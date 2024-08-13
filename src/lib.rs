//! A Bevy library for computing the Jump Flooding Algorithm.
//!
//! The **jump flooding algorithm** (JFA) is a fast screen-space algorithm for
//! computing distance fields. Currently, this crate provides a plugin for
//! adding outlines to arbitrary meshes.
//!
//! Outlines adapted from ["The Quest for Very Wide Outlines" by Ben Golus][0].
//!
//! [0]: https://bgolus.medium.com/the-quest-for-very-wide-outlines-ba82ed442cd9
//!
//! # Setup
//!
//! To add an outline to a mesh:
//!
//! 1. Add the [`OutlinePlugin`] to the base `App`.
//! 2. Add the desired [`OutlineStyle`] as an `Asset`.
//! 3. Add a [`CameraOutline`] component with the desired `OutlineStyle` to the
//!    camera which should render the outline.  Currently, outline styling is
//!    tied to the camera rather than the mesh.
//! 4. Add an [`Outline`] component to the mesh with `enabled: true`.

use std::ops::Range;

use bevy::{
    app::prelude::*,
    asset::{embedded_asset, Asset, AssetApp, Assets, Handle},
    color::Color,
    core_pipeline::core_3d,
    ecs::prelude::*,
    math::FloatOrd,
    pbr::{DrawMesh, MeshPipelineKey, MeshUniform, SetMeshBindGroup, SetMeshViewBindGroup},
    prelude::Camera3d,
    reflect::TypePath,
    render::{
        extract_resource::ExtractResource,
        prelude::*,
        render_asset::{RenderAssetPlugin, RenderAssets},
        render_graph::RenderGraph,
        render_phase::{
            AddRenderCommand, BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId,
            DrawFunctions, PhaseItem, PhaseItemExtraIndex, SetItemPipeline,
        },
        render_resource::*,
        view::{ExtractedView, VisibleEntities},
        Extract, RenderApp, RenderSet,
    },
};
use graph::OutlineDriverNodeLabel;

use crate::{graph::OutlineDriverNode, mask::MeshMaskPipeline, outline::GpuOutlineParams};

mod graph;
mod jfa;
mod jfa_init;
mod mask;
mod outline;
mod resources;

const JFA_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rg16Snorm;
const FULLSCREEN_PRIMITIVE_STATE: PrimitiveState = PrimitiveState {
    topology: PrimitiveTopology::TriangleList,
    strip_index_format: None,
    front_face: FrontFace::Ccw,
    cull_mode: Some(Face::Back),
    unclipped_depth: false,
    polygon_mode: PolygonMode::Fill,
    conservative: false,
};

/// Top-level plugin for enabling outlines.
#[derive(Default)]
pub struct OutlinePlugin;

/// Performance and visual quality settings for JFA-based outlines.
#[derive(Clone, ExtractResource, Resource)]
pub struct OutlineSettings {
    pub(crate) half_resolution: bool,
}

impl OutlineSettings {
    /// Returns whether the half-resolution setting is enabled.
    pub fn half_resolution(&self) -> bool {
        self.half_resolution
    }

    /// Sets whether the half-resolution setting is enabled.
    pub fn set_half_resolution(&mut self, value: bool) {
        self.half_resolution = value;
    }
}

impl Default for OutlineSettings {
    fn default() -> Self {
        Self {
            half_resolution: false,
        }
    }
}

const MASK_SHADER: &str = "shaders/mask.wgsl";
const JFA_INIT_SHADER: &str = "shaders/jfa_init.wgsl";
const JFA_SHADER: &str = "shaders/jfa.wgsl";
const FULLSCREEN_SHADER: &str = "shaders/fullscreen.wgsl";
const OUTLINE_SHADER: &str = "shaders/outline.wgsl";
const DIMENSIONS_SHADER: &str = "shaders/dimensions.wgsl";

use crate::graph::outline as outline_graph;

impl Plugin for OutlinePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RenderAssetPlugin::<GpuOutlineParams>::default())
            .init_asset::<OutlineStyle>()
            .init_resource::<OutlineSettings>();

        let mut shaders = app
            .world_mut()
            .get_resource_mut::<Assets<Shader>>()
            .unwrap();

        embedded_asset!(app, "shaders/mask.wgsl");
        embedded_asset!(app, "shaders/jfa_init.wgsl");
        embedded_asset!(app, "shaders/jfa.wgsl");
        embedded_asset!(app, "shaders/fullscreen.wgsl");
        embedded_asset!(app, "shaders/outline.wgsl");
        embedded_asset!(app, "shaders/dimensions.wgsl");

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Some(r) => r,
            None => return,
        };

        render_app
            .init_resource::<DrawFunctions<MeshMask>>()
            .add_render_command::<MeshMask, SetItemPipeline>()
            .add_render_command::<MeshMask, DrawMeshMask>()
            .init_resource::<resources::OutlineResources>()
            .init_resource::<mask::MeshMaskPipeline>()
            .init_resource::<SpecializedMeshPipelines<mask::MeshMaskPipeline>>()
            .init_resource::<jfa_init::JfaInitPipeline>()
            .init_resource::<jfa::JfaPipeline>()
            .init_resource::<outline::OutlinePipeline>()
            .init_resource::<SpecializedRenderPipelines<outline::OutlinePipeline>>()
            .add_systems(
                ExtractSchedule,
                (
                    extract_outline_settings,
                    extract_camera_outlines,
                    extract_mask_camera_phase,
                ),
            )
            .add_systems(
                Update,
                (resources::recreate_outline_resources, queue_mesh_masks).in_set(RenderSet::Queue),
            );

        let outline_graph = graph::outline(render_app).unwrap();

        let mut root_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        let draw_3d_graph = root_graph
            .get_sub_graph_mut(core_3d::graph::Core3d)
            .unwrap();
        let draw_3d_input = draw_3d_graph.input_node().label;

        draw_3d_graph.add_sub_graph(outline_graph::Name, outline_graph);
        draw_3d_graph.add_node(OutlineDriverNodeLabel, OutlineDriverNode);
        draw_3d_graph.add_slot_edge(
            draw_3d_input,
            core_3d::graph::input::VIEW_ENTITY,
            OutlineDriverNodeLabel,
            OutlineDriverNode::INPUT_VIEW,
        );
        draw_3d_graph.add_node_edge(
            // might be another pass
            core_3d::graph::Node3d::StartMainPass,
            OutlineDriverNodeLabel,
        );
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MeshMaskKey {
    distance: FloatOrd,
    pipeline: CachedRenderPipelineId,
    entity: Entity,
    draw_function: DrawFunctionId,
}

struct MeshMask {
    key: MeshMaskKey,
    pub representative_entity: Entity,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for MeshMask {
    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.key.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl CachedRenderPipelinePhaseItem for MeshMask {
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.key.pipeline
    }
}

impl BinnedPhaseItem for MeshMask {
    type BinKey = MeshMaskKey;

    fn new(
        key: Self::BinKey,
        representative_entity: Entity,
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        Self {
            key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

type DrawMeshMask = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    DrawMesh,
);

/// Visual style for an outline.
#[derive(Clone, Debug, PartialEq, TypePath, Asset)]
pub struct OutlineStyle {
    pub color: Color,
    pub width: f32,
}

/// Component for enabling outlines when rendering with a given camera.
#[derive(Clone, Debug, PartialEq, Component)]
pub struct CameraOutline {
    pub enabled: bool,
    pub style: Handle<OutlineStyle>,
}

/// Component for entities that should be outlined.
#[derive(Clone, Debug, PartialEq, Component)]
pub struct Outline {
    pub enabled: bool,
}

fn extract_outline_settings(mut commands: Commands, settings: Extract<Res<OutlineSettings>>) {
    commands.insert_resource(settings.clone());
}

fn extract_camera_outlines(
    mut commands: Commands,
    mut previous_outline_len: Local<usize>,
    cam_outline_query: Extract<Query<(Entity, &CameraOutline), With<Camera>>>,
) {
    let mut batches = Vec::with_capacity(*previous_outline_len);
    batches.extend(
        cam_outline_query
            .iter()
            .filter_map(|(entity, outline)| outline.enabled.then(|| (entity, (outline.clone(),)))),
    );
    *previous_outline_len = batches.len();
    commands.insert_or_spawn_batch(batches);
}

fn extract_mask_camera_phase(
    mut commands: Commands,
    cameras: Extract<Query<Entity, (With<Camera3d>, With<CameraOutline>)>>,
) {
    for entity in cameras.iter() {
        commands
            .get_or_spawn(entity)
            .insert(RenderPhase::<MeshMask>::default());
    }
}

fn queue_mesh_masks(
    mesh_mask_draw_functions: Res<DrawFunctions<MeshMask>>,
    mesh_mask_pipeline: Res<MeshMaskPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<MeshMaskPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    render_meshes: Res<RenderAssets<Mesh>>,
    outline_meshes: Query<(Entity, &Handle<Mesh>, &MeshUniform)>,
    mut views: Query<(
        &ExtractedView,
        &mut VisibleEntities,
        &mut RenderPhase<MeshMask>,
    )>,
) {
    let draw_outline = mesh_mask_draw_functions
        .read()
        .get_id::<DrawMeshMask>()
        .unwrap();

    for (view, visible_entities, mut mesh_mask_phase) in views.iter_mut() {
        let view_matrix = view.transform.compute_matrix();
        let inv_view_row_2 = view_matrix.inverse().row(2);

        for visible_entity in visible_entities.entities.iter().copied() {
            let (entity, mesh_handle, mesh_uniform) = match outline_meshes.get(visible_entity) {
                Ok(m) => m,
                Err(_) => continue,
            };

            let mesh = match render_meshes.get(mesh_handle) {
                Some(m) => m,
                None => continue,
            };

            let key = MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);

            let pipeline = pipelines
                .specialize(&mut pipeline_cache, &mesh_mask_pipeline, key, &mesh.layout)
                .unwrap();

            mesh_mask_phase.add(MeshMask {
                entity,
                pipeline,
                draw_function: draw_outline,
                distance: inv_view_row_2.dot(mesh_uniform.transform.col(3)),
            });
        }
    }
}
