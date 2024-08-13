use bevy::{
    asset::embedded_path,
    pbr::{MeshPipeline, MeshPipelineKey},
    prelude::*,
    render::{
        mesh::MeshVertexBufferLayoutRef,
        render_graph::{Node, RenderGraphContext, SlotInfo, SlotType},
        render_phase::ViewBinnedRenderPhases,
        render_resource::{
            ColorTargetState, ColorWrites, FragmentState, LoadOp, MultisampleState, Operations,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            SpecializedMeshPipeline, SpecializedMeshPipelineError, StoreOp, TextureFormat,
        },
        renderer::RenderContext,
    },
};

use crate::{resources::OutlineResources, MeshMask, MASK_SHADER};

#[derive(Resource)]
pub struct MeshMaskPipeline {
    mesh_pipeline: MeshPipeline,
    shader: Handle<Shader>,
}

impl FromWorld for MeshMaskPipeline {
    fn from_world(world: &mut World) -> Self {
        MeshMaskPipeline {
            mesh_pipeline: world.get_resource::<MeshPipeline>().unwrap().clone(),
            shader: world
                .get_resource::<AssetServer>()
                .unwrap()
                .load(embedded_path!(MASK_SHADER)),
        }
    }
}

impl SpecializedMeshPipeline for MeshMaskPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut desc = self.mesh_pipeline.specialize(key, layout)?;

        desc.layout = self
            .mesh_pipeline
            .view_layouts
            .iter()
            .map(|view| view.bind_group_layout.clone())
            .collect();

        desc.vertex.shader = self.shader.clone_weak();

        desc.fragment = Some(FragmentState {
            shader: self.shader.clone_weak(),
            shader_defs: vec![],
            entry_point: "fragment".into(),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::R8Unorm,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        });
        desc.depth_stencil = None;

        desc.multisample = MultisampleState {
            count: 4,
            mask: !0,
            alpha_to_coverage_enabled: false,
        };

        desc.label = Some("mesh_stencil_pipeline".into());
        Ok(desc)
    }
}

/// Render graph node for producing stencils from meshes.
pub struct MeshMaskNode;

impl MeshMaskNode {
    pub const IN_VIEW: &'static str = "view";

    /// The produced stencil buffer.
    ///
    /// This has format `TextureFormat::Depth24PlusStencil8`. Fragments covered
    /// by a mesh are assigned a value of 255. All other fragments are assigned
    /// a value of 0. The depth aspect is unused.
    pub const OUT_MASK: &'static str = "stencil";
}

impl Node for MeshMaskNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(Self::IN_VIEW, SlotType::Entity)]
    }

    fn output(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(Self::OUT_MASK, SlotType::TextureView)]
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let res = world.get_resource::<OutlineResources>().unwrap();
        let input_view_entity = graph.view_entity();

        let Some(mesh_mask_render_phases) =
            world.get_resource::<ViewBinnedRenderPhases<MeshMask>>()
        else {
            return Ok(());
        };

        let Some(stencil_phase) = mesh_mask_render_phases.get(&input_view_entity) else {
            return Ok(());
        };

        graph
            .set_output(Self::OUT_MASK, res.mask_multisample.default_view.clone())
            .unwrap();

        let mut tracked_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("outline_stencil_render_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &res.mask_multisample.default_view,
                resolve_target: Some(&res.mask_output.default_view),
                ops: Operations {
                    load: LoadOp::Clear(LinearRgba::BLACK.into()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        stencil_phase.render(&mut tracked_pass, world, input_view_entity);

        Ok(())
    }
}
