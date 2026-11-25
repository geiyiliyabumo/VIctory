use super::*;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

fn image(format: DXGI_FORMAT, flags: u32, state: D3D12_RESOURCE_STATES) -> ResourceRef {
    let desc = ResourceDesc {
        flags,
        ..texture(4, 4, format.0 as u32, 1)
    };
    let mut out = std::ptr::null();
    assert_eq!(
        create_committed(
            std::ptr::null(),
            &PROPERTIES,
            0,
            &desc,
            state.0 as u32,
            std::ptr::null(),
            &RESOURCE_IIDS[0],
            &mut out
        ),
        S_OK
    );
    ResourceRef(out as usize)
}

#[test]
fn replay_uses_render_target_and_depth_views_as_bound() {
    let Some(host) = crate::host::host() else {
        assert!(
            !crate::host::on_warp(),
            "requested software device unavailable"
        );
        return;
    };
    let colors = [0, 1].map(|_| {
        image(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            1,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        )
    });
    let depths = [0, 1].map(|_| {
        image(
            DXGI_FORMAT_D32_FLOAT,
            2 | 8,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        )
    });
    let color = colors
        .each_ref()
        .map(|r| host_resource(r.as_ptr()).unwrap());
    let depth = depths
        .each_ref()
        .map(|r| host_resource(r.as_ptr()).unwrap());
    let source = r#"
        [RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)")]
        float4 vs(uint id : SV_VertexID) : SV_Position {
            float2 p = float2((id << 1) & 2, id & 2);
            return float4(2 * p - 1, 0.5, 1);
        }
        float4 ps() : SV_Target { return float4(1, 0, 0, 1); }
    "#;
    let vs = crate::host::compile(source, s!("vs"), s!("vs_5_1")).unwrap();
    let ps = crate::host::compile(source, s!("ps"), s!("ps_5_1")).unwrap();
    // SAFETY: the VS embeds its root signature and both bytecode blobs remain alive.
    let root: ID3D12RootSignature = unsafe { host.device.CreateRootSignature(0, &vs) }.unwrap();
    let bytecode = |s: &[u8]| D3D12_SHADER_BYTECODE {
        pShaderBytecode: s.as_ptr().cast(),
        BytecodeLength: s.len(),
    };
    let mut pso = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
        VS: bytecode(&vs),
        PS: bytecode(&ps),
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            ..Default::default()
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    pso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    let pipeline: ID3D12PipelineState =
        unsafe { host.device.CreateGraphicsPipelineState(&pso) }.unwrap();
    // SAFETY: release the extra reference put into the ABI's ManuallyDrop field above.
    unsafe { std::mem::ManuallyDrop::drop(&mut pso.pRootSignature) };
    let rb = host.read_back_buffer(2048).unwrap();
    let mut results = Vec::new();
    for (replace_color, replace_depth) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let mut color_slot = [0u64; 16];
        let mut depth_slot = [0u64; 16];
        let c = color_slot.as_mut_ptr() as u64;
        let d = depth_slot.as_mut_ptr() as u64;
        let bind_records = |ci: usize, di: usize| {
            crate::descriptor::create_rtv(
                std::ptr::null(),
                colors[ci].as_ptr(),
                std::ptr::null(),
                c,
            );
            crate::descriptor::create_dsv(
                std::ptr::null(),
                depths[di].as_ptr(),
                std::ptr::null(),
                d,
            );
        };
        bind_records(0, 0);
        let first = host.new_list().unwrap();
        let mut state = crate::draw::State::default();
        state.set_targets(&first, &[c], d);
        // SAFETY: the two color/depth pairs have the requested native states.
        unsafe {
            for i in 0..2 {
                first
                    .list
                    .ClearRenderTargetView(host.rtv(Some(color[i]), None), &[0.; 4], None);
                first.list.ClearDepthStencilView(
                    host.dsv(depth[i], None),
                    D3D12_CLEAR_FLAG_DEPTH,
                    if i == 0 { 1.0 } else { 0.0 },
                    0,
                    None,
                );
            }
        }
        first.close();
        host.execute(&[&first]);
        // Rewriting either CPU descriptor is legal after the OM call. After a
        // runtime-inserted list cut the original bindings still govern the draw.
        bind_records(usize::from(replace_color), usize::from(replace_depth));
        let list = host.new_list().unwrap();
        state.replay(&list);
        // SAFETY: all shader state, buffers and descriptors outlive this submission.
        unsafe {
            list.list.SetGraphicsRootSignature(&root);
            list.list.SetPipelineState(&pipeline);
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 4.,
                Height: 4.,
                MaxDepth: 1.,
                ..Default::default()
            }]);
            list.list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: 4,
                bottom: 4,
            }]);
            list.list.DrawInstanced(3, 1, 0, 0);
            for (i, image) in color.iter().enumerate() {
                list.list.ResourceBarrier(&[crate::host::transition(
                    image,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let dest = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&rb),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 1024 * i as u64,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: 4,
                                Height: 4,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(*image),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                list.list.CopyTextureRegion(&dest, 0, 0, 0, &src, None);
                list.list.ResourceBarrier(&[crate::host::transition(
                    image,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                )]);
            }
        }
        list.close();
        host.execute(&[&list]);
        let mut mapping = std::ptr::null_mut();
        let mut red = [0; 2];
        // SAFETY: both images occupy known padded rows in the completed readback.
        unsafe {
            rb.Map(0, None, Some(&mut mapping)).unwrap();
            for (i, count) in red.iter_mut().enumerate() {
                for y in 0..4 {
                    for x in 0..4 {
                        let pt = mapping
                            .cast::<u8>()
                            .add(i * 1024 + y * 256 + x * 4)
                            .cast::<[u8; 4]>()
                            .read();
                        assert!(
                            matches!(pt, [0, 0, 0, 0] | [255, 0, 0, 255]),
                            "unexpected pixel {pt:?}"
                        );
                        *count += usize::from(pt[0] == 255);
                    }
                }
            }
            rb.Unmap(0, Some(&D3D12_RANGE::default()));
        }
        eprintln!("rewrite RTV={replace_color} DSV={replace_depth}: red pixels {red:?}");
        results.push(red);
    }
    assert_eq!(
        results,
        [[16, 0]; 4],
        "a rewritten CPU descriptor changed an established output binding"
    );
}
