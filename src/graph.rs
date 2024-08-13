use bevy::{
    prelude::*,
    render::{
        render_graph::{
            GraphInput, Node, NodeRunError, RenderGraph, RenderGraphContext, RenderGraphError,
            RenderLabel, SlotInfo, SlotType,
        },
        render_resource::TextureFormat,
        renderer::RenderContext,
        texture::BevyDefault,
    },
};

use crate::{jfa::JfaNode, jfa_init::JfaInitNode, mask::MeshMaskNode, outline::OutlineNode};

pub(crate) mod outline {
    use bevy::render::render_graph::RenderSubGraph;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, RenderSubGraph)]
    pub struct Name;

    pub mod input {
        pub const VIEW_ENTITY: &str = "view_entity";
    }

    pub mod node {
        use bevy::render::render_graph::RenderLabel;

        #[derive(Debug, Clone, PartialEq, Eq, Hash, RenderLabel)]
        pub struct MaskPass;
        #[derive(Debug, Clone, PartialEq, Eq, Hash, RenderLabel)]
        pub struct JfaInitPass;
        #[derive(Debug, Clone, PartialEq, Eq, Hash, RenderLabel)]
        pub struct JfaPass;
        #[derive(Debug, Clone, PartialEq, Eq, Hash, RenderLabel)]
        pub struct OutlinePass;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, RenderLabel)]
pub struct OutlineDriverNodeLabel;

pub struct OutlineDriverNode;

impl OutlineDriverNode {
    pub const INPUT_VIEW: &'static str = "view_entity";
}

impl Node for OutlineDriverNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        _render_context: &mut RenderContext,
        _world: &World,
    ) -> Result<(), NodeRunError> {
        let view_ent = graph.get_input_entity(Self::INPUT_VIEW)?;

        graph.run_sub_graph(outline::Name, vec![], Some(view_ent))?;

        Ok(())
    }

    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo {
            name: Self::INPUT_VIEW.into(),
            slot_type: SlotType::Entity,
        }]
    }
}

/// Builds the render graph for applying the JFA outline.
pub fn outline(render_app: &mut SubApp) -> Result<RenderGraph, RenderGraphError> {
    let mut graph = RenderGraph::default();

    graph.set_input(vec![SlotInfo {
        name: outline::input::VIEW_ENTITY.into(),
        slot_type: SlotType::Entity,
    }]);

    // Graph order:
    // 1. Mask
    // 2. JFA Init
    // 3. JFA
    // 4. Outline

    let mask_node = MeshMaskNode;
    let jfa_node = JfaNode::from_world(render_app.world_mut());
    // TODO: BevyDefault for surface texture format is an anti-pattern;
    // the target texture format should be queried from the window when
    // Bevy exposes that functionality.
    let outline_node = OutlineNode::new(render_app.world_mut(), TextureFormat::bevy_default());

    graph.add_node(outline::node::MaskPass, mask_node);
    graph.add_node(outline::node::JfaInitPass, JfaInitNode);
    graph.add_node(outline::node::JfaPass, jfa_node);
    graph.add_node(outline::node::OutlinePass, outline_node);

    // Input -> Mask
    graph.add_slot_edge(
        GraphInput,
        outline::input::VIEW_ENTITY,
        outline::node::MaskPass,
        MeshMaskNode::IN_VIEW,
    );

    // Mask -> JFA Init
    graph.add_slot_edge(
        outline::node::MaskPass,
        MeshMaskNode::OUT_MASK,
        outline::node::JfaInitPass,
        JfaInitNode::IN_MASK,
    );

    // Input -> JFA
    graph.add_slot_edge(
        GraphInput,
        outline::input::VIEW_ENTITY,
        outline::node::JfaPass,
        JfaNode::IN_VIEW,
    );

    // JFA Init -> JFA
    graph.add_slot_edge(
        outline::node::JfaInitPass,
        JfaInitNode::OUT_JFA_INIT,
        outline::node::JfaPass,
        JfaNode::IN_BASE,
    );

    // Input -> Outline
    graph.add_slot_edge(
        GraphInput,
        outline::input::VIEW_ENTITY,
        outline::node::OutlinePass,
        OutlineNode::IN_VIEW,
    );

    // JFA -> Outline
    graph.add_slot_edge(
        outline::node::JfaPass,
        JfaNode::OUT_JUMP,
        outline::node::OutlinePass,
        OutlineNode::IN_JFA,
    );

    Ok(graph)
}
