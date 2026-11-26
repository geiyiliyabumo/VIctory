//! Guest memory by address, including addresses read from scalar descriptors.
use super::{Attr, Bin, Cast, Fields, Inst, Lower, NULL, Operand, Reg, RootType, Site, Tag, Term, V, Why, pred};

impl Lower<'_> {
    /// Some ray shaders take a four-word descriptor from a computed 32-byte
    /// heap slot and use only its low address and masked high address bits.
    /// A numeric use of any format, stride or record-count bits is a different ABI.
    pub(super) fn picked_address_words(&self, site: Site, code: u8) -> bool {
        let Some(a) = self.scalar.get(&site) else {
            return false;
        };
        let Some(dynamic) = a.dynamic else {
            return false;
        };
        let Some(block) = self.def_block.get(&dynamic) else {
            return false;
        };
        let aligned = self.x.p.blocks[*block]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == dynamic))
            .is_some_and(|i| {
                i.name() == "s_lshl_b32"
                    && matches!(&i.fields, Fields::Sop {src, ..}
                if matches!(src.as_slice(), [_, Operand::Const(133) | Operand::Literal(5)]))
            });
        if !aligned
            || !a.offset.is_multiple_of(32)
            || !self
                .x
                .native
                .layout
                .get(a.param)
                .is_some_and(|p| p.root == RootType::ResourceHeap)
        {
            return false;
        }
        let loaded = &self.x.p.blocks[site.0].insts[site.1];
        let (Some(lo), Some(hi)) = (loaded.write(Reg::S(code)), loaded.write(Reg::S(code + 1)))
        else {
            return false;
        };
        let other = [
            loaded.write(Reg::S(code + 2)),
            loaded.write(Reg::S(code + 3)),
        ];
        let (mut masked, mut aligned_pointer) = (false, false);
        for block in &self.x.p.blocks {
            if block
                .phis
                .iter()
                .any(|p| p.args.iter().any(|v| *v == hi || other.contains(&Some(*v))))
            {
                return false;
            }
            for i in &block.insts {
                if i.reads.iter().any(|r| r.1 == lo)
                    && i.name() == "s_and_b32"
                    && matches!(&i.fields, Fields::Sop {src, ..} if src.contains(&Operand::Literal(0xffff_ff00)))
                {
                    aligned_pointer = true;
                }
                if i.reads.iter().any(|r| other.contains(&Some(r.1))) {
                    return false;
                }
                if !i.reads.iter().any(|r| r.1 == hi) {
                    continue;
                }
                if i.name() != "s_and_b32"
                    || !matches!(&i.fields, Fields::Sop {src, ..}
                    if src.iter().any(|o| matches!(o, Operand::Literal(v) if *v != 0 && *v <= 0xffff)))
                {
                    return false;
                }
                masked = true;
            }
        }
        masked && aligned_pointer
    }

    pub(super) fn picked_address(&mut self, site: Site) -> Result<(V, V), Why> {
        let slot = self.heap_index(site, 0, false)?;
        let c = self.cur;
        let handle = self.s.root_srv_handle(crate::emit::SLOT_VIEW_ADDRESSES);
        let undef = self.s.m.undef(self.s.t.i32);
        let zero = self.s.m.i32(0);
        let one = self.s.m.i32(1);
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let align = self.s.m.i32(4);
        let count_mask = self.s.m.i8(1);
        let address_mask = self.s.m.i8(3);
        let count = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[handle, zero, undef, count_mask, align],
        );
        let count = self.s.m.extract(c, count, 0);
        let count = self.as_int(count);
        let inside = self.s.m.cmp(c, pred::IULT, slot, count);
        let record = self.s.m.bin(c, Bin::Add, slot, two);
        let record = self.s.m.select(c, inside, record, one);
        let byte = self.s.m.bin(c, Bin::Shl, record, three);
        let words = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[handle, byte, undef, address_mask, align],
        );
        let lo = self.s.m.extract(c, words, 0);
        let hi = self.s.m.extract(c, words, 1);
        Ok((self.as_int(lo), self.as_int(hi)))
    }

    /// `lo`, `hi` plus the instruction's byte offset `off`, with the carry.
    pub(super) fn address_plus(&mut self, lo: V, hi: V, off: i32) -> (V, V) {
        if off == 0 {
            return (lo, hi);
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let k = self.s.m.i32(off as u32);
        let sum = self.s.m.bin(c, Bin::Add, lo, k);
        // Up: a carry when the sum wrapped below `lo`. Down: a borrow when
        // it wrapped above.
        let wrapped = self
            .s
            .m
            .cmp(c, if off > 0 { pred::IULT } else { pred::IUGT }, sum, lo);
        let one = self.s.m.cast(c, Cast::ZExt, wrapped, i32t);
        let hi = self.s.m.bin(c, if off > 0 { Bin::Add } else { Bin::Sub }, hi, one);
        (sum, hi)
    }

    /// Title memory at the 40-bit address `lo`, `hi` (`shader.rs`
    /// `SLOT_MEMORY`): the handle of the raw view of the host copy holding
    /// it, the byte in that view, and whether the page is mapped. A page without a copy reads zeros
    /// (and drops writes) through the null view at the first slot, and its
    /// number is left for the host, for lanes that run (`guard`).
    pub(super) fn memory(&mut self, lo: V, hi: V, guard: Option<V>) -> (V, V, V) {
        use super::super::shader::{MEMORY_FAULTS, MEMORY_PAGES, SLOT_FAULTS, SLOT_MEMORY, USER_MEMORY_BASE};
        let c = self.cur;
        let (i32t, res) = (self.s.t.i32, self.s.t.res_ret);
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k16, k8) = (k(self, 16), k(self, 0xFF));
        let low = self.s.m.bin(c, Bin::LShr, lo, k16);
        let high = self.s.m.bin(c, Bin::And, hi, k8);
        let high = self.s.m.bin(c, Bin::Shl, high, k16);
        let page = self.s.m.bin(c, Bin::Or, low, high);
        // Inside the directory whatever the address (lanes that do not run
        // compute one too); its last entry is never filled.
        let last = k(self, MEMORY_PAGES - 1);
        let page = self
            .s
            .op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[page, last]);
        self.s.m.tag(page, Tag::Clamp);
        let k3 = k(self, 3);
        let at = self.s.m.bin(c, Bin::Shl, page, k3);
        let directory = self.s.root_srv_handle(SLOT_MEMORY);
        let (iu, both, align) = (self.s.m.undef(i32t), self.s.m.i8(3), k(self, 4));
        let entry = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            res,
            139,
            &[directory, at, iu, both, align],
        );
        let (copy, start) = (self.s.m.extract(c, entry, 0), self.s.m.extract(c, entry, 1));
        let (copy, start) = (self.as_int(copy), self.as_int(start));
        // A page without a copy: its number, at a place by the number, or
        // at the sink for lanes that do not run and pages that have one.
        let zero = k(self, 0);
        // A newer command list can publish more pages after this one has
        // recorded. Only reference descriptors prepared for this particular bind.
        let count = self.host_word(RootType::Other(USER_MEMORY_BASE), 0, 1);
        let in_range = self.s.m.cmp(c, pred::IULE, copy, count);
        let copy = self.s.m.select(c, in_range, copy, zero);
        let mut missing = self.s.m.cmp(c, pred::EQ, copy, zero);
        if let Some(g) = guard {
            missing = self.s.m.bin(c, Bin::And, missing, g);
        }
        let places = k(self, MEMORY_FAULTS - 1);
        let place = self.s.m.bin(c, Bin::URem, page, places);
        let one = k(self, 1);
        let place = self.s.m.bin(c, Bin::Add, place, one);
        let k2 = k(self, 2);
        let place = self.s.m.bin(c, Bin::Shl, place, k2);
        let place = self.s.m.select(c, missing, place, zero);
        let faults = self.s.root_uav_handle(SLOT_FAULTS);
        let void = self.s.t.void;
        let mask = self.s.m.i8(1);
        self.s.op(
            c,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[faults, place, iu, page, iu, iu, iu, mask, align],
        );
        let base = self.host_word(RootType::Other(USER_MEMORY_BASE), 0, 0);
        let slot = self.s.m.bin(c, Bin::Add, base, copy);
        let h = self.s.heap_handle(c, slot, false, true, (11 | 0x1000, 0));
        let k64k = k(self, 0xFFFF);
        let within = self.s.m.bin(c, Bin::And, lo, k64k);
        let byte = self.s.m.bin(c, Bin::Add, start, within);
        let mapped = self.s.m.cmp(c, pred::NE, copy, zero);
        (h, byte, mapped)
    }

    /// `global_load_*`, `global_store_*` and `global_atomic_*` on title
    /// memory by address (`resource::MEMORY`): the address is the VGPR
    /// pair, each lane its own.
    pub(super) fn memory_flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            offset,
            lds,
            cache,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *lds {
            return Err("a global address in LDS");
        }
        let Some(Operand::Vector { reg: areg, .. }) = *addr else {
            return Err("a global address not in VGPRs");
        };
        let word = |l: &mut Self, r: Reg| -> Result<V, Why> {
            let v = inst.read(r).ok_or("a global address")?;
            l.i32_of(v)
        };
        let (lo, hi) = match *saddr {
            Some(Operand::Scalar { code, .. }) if code != NULL => {
                let base = word(self, Reg::S(code))?;
                let high = word(self, Reg::S(code + 1))?;
                let offset = word(self, Reg::V(areg))?;
                let c = self.cur;
                let sum = self.s.m.bin(c, Bin::Add, base, offset);
                let carry = self.s.m.cmp(c, pred::IULT, sum, base);
                let carry = self.s.m.cast(c, Cast::ZExt, carry, self.s.t.i32);
                (sum, self.s.m.bin(c, Bin::Add, high, carry))
            }
            _ => (word(self, Reg::V(areg))?, word(self, Reg::V(areg + 1))?),
        };
        let (lo, hi) = self.address_plus(lo, hi, i32::from(*offset));
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (i32t, res) = (self.s.t.i32, self.s.t.res_ret);
        let iu = self.s.m.undef(i32t);
        let align = self.s.m.i32(4);
        let name = inst.name();
        let count = |rest: &str| match rest {
            "dword" => Ok(1u8),
            "dwordx2" => Ok(2),
            "dwordx3" => Ok(3),
            "dwordx4" => Ok(4),
            _ => Err(name),
        };
        if let Some(rest) = name.strip_prefix("global_load_") {
            let n = count(rest)?;
            let Some(Operand::Vector { reg, .. }) = *dst else {
                return Err("a global load result");
            };
            let (h, byte, _) = self.memory(lo, hi, guard);
            let mask = self.s.m.i8((1u8 << n) - 1);
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                res,
                139,
                &[h, byte, iu, mask, align],
            );
            for k in 0..n {
                let x = self.s.m.extract(c, r, u32::from(k));
                let x = self.as_int(x);
                self.write(inst, Reg::V(reg + k), x, guard);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("global_store_") {
            let n = count(rest)?;
            let d = data.as_ref().ok_or("global store data")?;
            let vals = self.vgprs(inst, d, n)?;
            let (h, byte, _) = self.memory(lo, hi, guard);
            let at = self.begin_guarded(guard);
            if n == 1 {
                self.store_memory(h, byte, &vals);
            } else {
                // The directory maps pages independently. Keep the single
                // wide store inside a page and look up each later dword of
                // a straddling store at its own address, including the carry.
                let c = self.cur;
                let mask = self.s.m.i32(0xFFFF);
                let within = self.s.m.bin(c, Bin::And, lo, mask);
                let limit = self.s.m.i32(0x10000 - 4 * u32::from(n));
                let fits = self.s.m.cmp(c, pred::IULE, within, limit);
                let (plain, split, join) = (
                    self.s.m.new_block(),
                    self.s.m.new_block(),
                    self.s.m.new_block(),
                );
                self.s.m.terminate(c, Term::CondBr(fits, plain, split));
                self.cur = plain;
                self.store_memory(h, byte, &vals);
                self.s.m.terminate(plain, Term::Br(join));
                self.cur = split;
                self.store_memory(h, byte, &vals[..1]);
                for (k, value) in (1..n).zip(&vals[1..]) {
                    let (lo, hi) = self.address_plus(lo, hi, 4 * i32::from(k));
                    let (handle, byte, _) = self.memory(lo, hi, None);
                    self.store_memory(handle, byte, std::slice::from_ref(value));
                }
                self.s.m.terminate(split, Term::Br(join));
                self.cur = join;
            }
            self.end_guarded(at, &[]);
            self.s.writes_guest_memory = true;
            return Ok(());
        }
        let Some(op) = name.strip_prefix("global_atomic_") else {
            return Err(name);
        };
        let d = data.as_ref().ok_or("global atomic data")?;
        let v = self.vgprs(inst, d, 1)?[0];
        // With `glc` the value from before comes back, in the result
        // register the instruction names.
        let back = match (*dst, cache.glc) {
            (Some(Operand::Vector { reg, .. }), true) => Some(reg),
            _ => None,
        };
        let old = match back.and_then(|reg| inst.read(Reg::V(reg))) {
            Some(x) => self.i32_of(x)?,
            None => self.s.m.i32(0),
        };
        let (h, byte, _) = self.memory(lo, hi, guard);
        // DXIL's atomic operations: add 0, and 1, or 2, xor 3, signed
        // minimum 4 and maximum 5, unsigned 6 and 7, exchange 8.
        let simple = match op {
            "add" => Some((0, v)),
            "sub" => {
                let zero = self.s.m.i32(0);
                Some((0, self.s.m.bin(c, Bin::Sub, zero, v)))
            }
            "and" => Some((1, v)),
            "or" => Some((2, v)),
            "xor" => Some((3, v)),
            "smin" => Some((4, v)),
            "smax" => Some((5, v)),
            "umin" => Some((6, v)),
            "umax" => Some((7, v)),
            "swap" => Some((8, v)),
            "fmin" | "fmax" => None,
            _ => return Err(name),
        };
        let at = self.begin_guarded(guard);
        let was = match simple {
            Some((code, x)) => {
                let code = self.s.m.i32(code);
                self.s.op(
                    self.cur,
                    "atomicBinOp.i32",
                    Attr::NoUnwind,
                    i32t,
                    78,
                    &[h, code, byte, iu, iu, x],
                )
            }
            None => {
                // A float minimum or maximum has no atomic operation on
                // the host: the dword is exchanged for the smaller (larger)
                // of itself and the value until no other thread got in
                // between. Each failed pass means another one's succeeded;
                // the limit only keeps a broken card from hanging.
                let head = self.cur;
                let mask = self.s.m.i8(1);
                let first = self.s.op(
                    head,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    res,
                    139,
                    &[h, byte, iu, mask, align],
                );
                let first = self.s.m.extract(head, first, 0);
                let first = self.as_int(first);
                let (pass, done) = (self.s.m.new_block(), self.s.m.new_block());
                self.s.m.terminate(head, Term::Br(pass));
                self.cur = pass;
                let seen = self.s.m.phi(pass, i32t);
                let n = self.s.m.phi(pass, i32t);
                let (f32t, fv) = (self.s.t.f32, self.as_float(v));
                let fseen = self.as_float(seen);
                // `dx.op.binary`: FMax 35, FMin 36.
                let pick = if op == "fmin" { 36 } else { 35 };
                let wanted = self
                    .s
                    .op(pass, "binary.f32", Attr::ReadNone, f32t, pick, &[fseen, fv]);
                let wanted = self.as_int(wanted);
                let found = self.s.op(
                    pass,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &[h, byte, iu, iu, seen, wanted],
                );
                let one = self.s.m.i32(1);
                let n1 = self.s.m.bin(pass, Bin::Add, n, one);
                let lost = self.s.m.cmp(pass, pred::NE, found, seen);
                let limit = self.s.m.i32(1 << 16);
                let more = self.s.m.cmp(pass, pred::IULT, n1, limit);
                let again = self.s.m.bin(pass, Bin::And, lost, more);
                self.s.m.terminate(pass, Term::CondBr(again, pass, done));
                let zero = self.s.m.i32(0);
                self.s.m.set_phi(seen, vec![(first, head), (found, pass)]);
                self.s.m.set_phi(n, vec![(zero, head), (n1, pass)]);
                self.cur = done;
                found
            }
        };
        let was = self.end_guarded(at, &[(was, old)])[0];
        self.s.writes_guest_memory = true;
        if let Some(reg) = back {
            self.write(inst, Reg::V(reg), was, None);
        }
        Ok(())
    }

    fn store_memory(&mut self, handle: V, byte: V, values: &[V]) {
        let iu = self.s.m.undef(self.s.t.i32);
        let mut args = vec![handle, byte, iu];
        args.extend(values);
        args.resize(7, iu);
        args.push(self.s.m.i8((1u8 << values.len()) - 1));
        args.push(self.s.m.i32(4));
        self.s.op(
            self.cur,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            self.s.t.void,
            140,
            &args,
        );
    }

    /// `s_load_dword*` from title memory by address (`resource::MEMORY`):
    /// the SGPR pair plus the instruction's offsets.
    pub(super) fn memory_smem(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            base: Operand::Scalar { code: base, .. },
            offset,
            soffset,
            ..
        } = inst.fields
        else {
            return Err("a scalar load from memory");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u8,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory access by address"),
        };
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::S(base + k)).ok_or("an address not in SGPRs")?;
            l.i32_of(v)
        };
        let (mut lo, mut hi) = (word(self, 0)?, word(self, 1)?);
        let c = self.cur;
        let (i32t, res) = (self.s.t.i32, self.s.t.res_ret);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            // An offset register counts up from the address.
            let so = self.operand(inst, &soffset)?;
            let sum = self.s.m.bin(c, Bin::Add, lo, so);
            let carry = self.s.m.cmp(c, pred::IULT, sum, lo);
            let carry = self.s.m.cast(c, Cast::ZExt, carry, i32t);
            hi = self.s.m.bin(c, Bin::Add, hi, carry);
            lo = sum;
        }
        let (iu, align) = (self.s.m.undef(i32t), self.s.m.i32(4));
        for first in (0..count).step_by(4) {
            let n = (count - first).min(4);
            let (lo, hi) = self.address_plus(lo, hi, offset + 4 * i32::from(first));
            let (h, byte, _) = self.memory(lo, hi, None);
            let mask = self.s.m.i8((1u8 << n) - 1);
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                res,
                139,
                &[h, byte, iu, mask, align],
            );
            for k in 0..n {
                let x = self.s.m.extract(c, r, u32::from(k));
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + first + k), x, None);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::Lowered;
    use super::super::tests::{S_ENDPGM, lower_compute, lower_with, root};
    use crate::native::{HwStage, ResourceType, RootType, UsedResource};

    #[cfg(windows)]
    #[test]
    fn a_guest_store_crossing_4gib_uses_both_pages_and_reports_a_missing_page() {
        use crate::emit::host::{Dispatch, Host, Root, View};
        use crate::emit::{HostShader, MEMORY_FAULTS, USER_MEMORY_BASE, validator, valu_tests};
        use crate::native::RootParam;
        let words = [
            0x7E00_0280, // v_mov_b32 v0, 0 (byte offset from s[0:1])
            0x7E02_0281, // v_mov_b32 v1, 1
            0x7E04_0282, // v_mov_b32 v2, 2
            0x7E06_0283, // v_mov_b32 v3, 3
            0x7E08_0284, // v_mov_b32 v4, 4
            0xDC78_8000, 0x0000_0100, // global_store_dwordx4 v0, v[1:4], s[0:1]
            S_ENDPGM,
        ];
        let layout = vec![RootParam { sgprs: 2, ..root(RootType::Constants, 0, 0) }];
        let lowered = lower_compute(&words, layout, Vec::new(), false, false).unwrap();
        assert!(lowered.writes_guest_memory);
        let container = crate::Container::parse(&lowered.dxil).unwrap();
        let features = container.part(crate::FourCC::SFI0).unwrap().data.try_into().unwrap();
        assert_eq!(u64::from_le_bytes(features) & 0x8000, 0);
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            written_root_uavs: lowered.written_root_uavs,
            writes_guest_memory: lowered.writes_guest_memory,
            root_indexed_views: Vec::new(),
        };

        if let Ok(mut validator) = validator::Validator::load(
            validator::Validator::default_path().as_deref(),
        ) {
            validator.validate(&mut shader.dxil.clone()).unwrap();
        }

        let _gpu = valu_tests::ONE_AT_A_TIME
            .lock().unwrap_or_else(|e| e.into_inner());
        let host = Host::new(true).expect("WARP host");
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        for (first, second, lo, high) in [
            (0xFFFF, 0x1_0000, 0xFFFF_FFF8, 0),
            (0x0FFF, 0x1000, 0x0FFF_FFF8, 0),
        ] {
            for mapped in [true, false] {
                let mut directory = vec![0; (second + 1) * 2];
                directory[first * 2..first * 2 + 2].copy_from_slice(&[2, 0]);
                if mapped {
                    directory[second * 2..second * 2 + 2].copy_from_slice(&[1, 16]);
                }
                let mut user = vec![0; shader.user_data_dwords() as usize];
                for item in &shader.user_data {
                    user[item.index as usize] = match (item.root, item.dword) {
                        (RootType::Constants, 0) => lo,
                        (RootType::Constants, 1) => high,
                        (RootType::Other(USER_MEMORY_BASE), 0) => 0,
                        (RootType::Other(USER_MEMORY_BASE), 1) => 2,
                        _ => panic!("unexpected user data {item:?}"),
                    };
                }
                let out = host.dispatch(&root, &pso, &Dispatch {
                    roots: vec![
                        (0, Root::Constants(user)),
                        (1, Root::Uav(3)),
                        (2, Root::View { srv: true, data: directory }),
                    ],
                    views: &[
                        View::RawUav(1),
                        View::RawUav(16),
                        View::RawUav(0x4000),
                        View::RawUav(MEMORY_FAULTS),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                }).unwrap();
                assert_eq!(&out[2][0x3FFE..], &[1, 2]);
                assert_eq!(&out[1][4..6], if mapped { &[3, 4] } else { &[0, 0] });
                let faults: Vec<_> = out[3][1..].iter().copied().filter(|&page| page != 0).collect();
                assert_eq!(faults, if mapped { vec![] } else { vec![second as u32] });
            }
        }
    }

    #[test]
    fn dynamic_address_only_descriptor_reads_use_the_guest_address_table() {
        let words = [
            0xBE84_0300,
            0xBE85_0380, // s[4:5] = resource heap
            0x8F02_8501, // s2 = root index s1 << 5
            0xF408_0202,
            0x0400_0000, // s_load_dwordx4 s[8:11], s[4:5], s2
            0x870C_FF08,
            0xFFFF_FF00, // low address bits
            0x870D_FF09,
            0x0000_FFFF, // high address bits
            0x7E02_020C,
            0x7E04_020D, // v1 = s12, v2 = s13
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0xD834_0004,
            0x0000_0200, // ds_write_b32 v0, v2 offset:4
            S_ENDPGM,
        ];
        let compile = |code: &[u32]| {
            let layout = vec![
                root(RootType::ResourceHeap, 0, 0),
                root(RootType::Constants, 1, 1),
            ];
            lower_with(code, layout, vec![])
        };
        let requires_addresses =
            |shader: Lowered| shader.root_srvs.contains(&crate::emit::SLOT_VIEW_ADDRESSES);
        assert!(requires_addresses(compile(&words).unwrap()));
        let mut uses_stride = words;
        uses_stride[8] = 0xFFFF_0000; // the upper half of word 1 is descriptor metadata
        assert!(!compile(&uses_stride).is_ok_and(requires_addresses));
        let mut uses_record_count = words;
        uses_record_count[9] = 0x7E02_020A; // v1 = s10, word 2
        assert!(!compile(&uses_record_count).is_ok_and(requires_addresses));
    }

    #[test]
    #[cfg(windows)]
    fn warp_executes_guest_heap_address_selection_and_bounds_check() {
        use crate::emit::host::{Dispatch, Host, Root, View};
        let words = [
            0xBE84_0300,
            0xBE85_0380, // s[4:5] = resource heap
            0xBE86_0302,
            0xBE87_0380, // s[6:7] = output table
            0x8F02_8501, // s2 = root index s1 << 5
            0xF408_0202,
            0x0400_0000, // s[8:11] = heap[index]
            0x870C_FF08,
            0xFFFF_FF00, // s12 = low address aligned to 256
            0x870D_FF09,
            0x0000_FFFF, // s13 = high 16 address bits
            0x7E02_020C,
            0x7E04_020D, // v1 = s12, v2 = s13
            0xF408_0403,
            0xFA00_0000, // s[16:19] = output UAV
            0xE070_1000,
            0x8004_0100, // buffer_store_dword v1, v0, s[16:19], 0 offen
            0xE070_1004,
            0x8004_0200, // buffer_store_dword v2, v0, s[16:19], 0 offen offset:4
            S_ENDPGM,
        ];
        let layout = vec![
            root(RootType::ResourceHeap, 0, 0),
            root(RootType::Constants, 1, 1),
            root(RootType::Table, 2, 2),
        ];
        let output = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 2,
            slot: 2,
        };
        let lowered = lower_compute(&words, layout, vec![output], false, false).unwrap();
        let shader = crate::emit::HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            written_root_uavs: lowered.written_root_uavs,
            writes_guest_memory: lowered.writes_guest_memory,
            root_indexed_views: Vec::new(),
        };
        assert_eq!(shader.root_srvs, [crate::emit::SLOT_VIEW_ADDRESSES]);
        let _one = crate::emit::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(true) else { return };
        let (sig, pso) = host.compute_pipeline(&shader).unwrap();
        for (index, expected) in [
            (0, [0x1122_3300, 13]),
            (1, [0x5566_7700, 14]),
            (2, [0, 0]),
            (99, [0, 0]),
        ] {
            let mut data = vec![0; shader.user_data_dwords().max(1) as usize];
            for param in &shader.user_data {
                if param.root == RootType::Constants {
                    data[param.index as usize] = index;
                }
            }
            let addresses = vec![2, 0, 0, 0, 0x1122_3300, 13, 0x5566_7700, 14];
            let outputs = host
                .dispatch(
                    &sig,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(data)),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: addresses,
                                },
                            ),
                        ],
                        views: &[View::RawUav(2)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(outputs[0], expected, "heap index {index}");
        }
    }
}
