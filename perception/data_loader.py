import glob
import io
import os
import argparse
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from PIL import Image

PERCEPTION_DIR = Path(__file__).resolve().parent
SMARTGRASP_ROOT = PERCEPTION_DIR.parent
DATA_DIR = Path(os.environ.get('SMARTGRASP_DATA_DIR', SMARTGRASP_ROOT / 'data')).expanduser()
PARQUET_GLOB = str(DATA_DIR / '*.parquet')
NPZ_DIR = DATA_DIR / 'npz_file'
IMAGE_SAMPLE_DIR = DATA_DIR / 'image_samples'
NPZ_VIZ_DIR = DATA_DIR / 'npz_visualizations'


def ensure_analysis_output_dirs():
    """Create optional data-analysis output folders on demand."""
    os.makedirs(IMAGE_SAMPLE_DIR, exist_ok=True)
    os.makedirs(NPZ_VIZ_DIR, exist_ok=True)


def format_size(size_bytes):
    """把文件大小格式化成便于阅读的字符串。"""
    if size_bytes < 1024:
        return f'{size_bytes} B'
    if size_bytes < 1024 * 1024:
        return f'{size_bytes / 1024:.2f} KB'
    return f'{size_bytes / (1024 * 1024):.2f} MB'


def describe_array(arr):
    """返回 npz 数组的形状、类型和简单数值统计。"""
    shape = getattr(arr, 'shape', None)
    dtype = getattr(arr, 'dtype', None)
    desc = f'shape={shape}, dtype={dtype}'

    if isinstance(arr, np.ndarray) and arr.size and np.issubdtype(arr.dtype, np.number):
        finite = arr[np.isfinite(arr)]
        if finite.size:
            desc += f', min={finite.min():.4g}, max={finite.max():.4g}, mean={finite.mean():.4g}'
    elif isinstance(arr, np.ndarray) and arr.size:
        sample = arr.flat[0]
        desc += f', sample={str(sample)[:80]}'

    return desc

def analyze_data_directory():
    """分析 ./data 目录结构"""
    ensure_analysis_output_dirs()
    print('='*60)
    print(f'[1] data 目录结构分析: {DATA_DIR}')
    print('='*60)

    if not DATA_DIR.exists():
        print(f'数据目录不存在: {DATA_DIR}')
        return
    
    # 检查目录内容
    for path in sorted(DATA_DIR.iterdir()):
        if path.is_file():
            print(f'  文件: {path.name} ({format_size(path.stat().st_size)})')
        elif path.is_dir():
            file_count = sum(1 for p in path.rglob('*') if p.is_file())
            print(f'  文件夹: {path.name}/ (包含 {file_count} 个文件)')
    
    # 检查 npz 压缩包
    npz_zip = DATA_DIR / 'npz_file.zip'
    if npz_zip.exists() and not NPZ_DIR.exists():
        print('\n[提示] 发现 npz_file.zip 但未解压，检查压缩包内容...')
        try:
            with zipfile.ZipFile(npz_zip, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f'  压缩包内有 {len(file_list)} 个文件（前10个）:')
                for f in file_list[:10]:
                    print(f'    - {f}')
        except Exception as e:
            print(f'  读取压缩包失败: {e}')


def analyze_parquet_files(example_count=3):
    """加载并分析 data 目录下的 parquet 文件。"""
    ensure_analysis_output_dirs()
    print('\n' + '='*60)
    print('[2] Parquet 数据集分析')
    print('='*60)

    parquet_files = glob.glob(PARQUET_GLOB)
    if not parquet_files:
        print(f'未找到 {PARQUET_GLOB} 下的 parquet 文件。')
        return None

    print(f'发现 {len(parquet_files)} 个 parquet 文件，文件大小:')
    for pf in sorted(parquet_files):
        print(f'  - {os.path.basename(pf)}: {format_size(os.path.getsize(pf))}')
    
    print('合并读取中...')
    dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
    df = pd.concat(dfs, ignore_index=True)

    print('\n[基本信息]')
    print(f'样本总数: {len(df)}')
    print(f'内存占用: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB')
    print(f'字段数: {len(df.columns)}')
    print('字段列表:', list(df.columns))

    print('\n[每个字段内容类型分析]')
    for col in df.columns:
        sample = df[col].iloc[0]
        dtype = type(sample).__name__
        # 特殊处理字典/列表类型
        if isinstance(sample, dict):
            print(f' - {col}: 类型={dtype}，键={list(sample.keys())}')
        elif isinstance(sample, list):
            print(f' - {col}: 类型={dtype}，长度={len(sample)}')
        else:
            print(f' - {col}: 类型={dtype}，样例={str(sample)[:80]}')

    print('\n[字段分布统计]')
    if 'sceneId' in df.columns:
        unique_scenes = df['sceneId'].nunique()
        print(f"场景总数: {unique_scenes}")
        print(f"  样例场景ID: {df['sceneId'].unique()[:5].tolist()}")
        samples_per_scene = df['sceneId'].value_counts()
        print(f"  每个场景的样本数 - 平均: {samples_per_scene.mean():.1f}, 最小: {samples_per_scene.min()}, 最大: {samples_per_scene.max()}")
    
    if 'queryObjId' in df.columns:
        unique_objs = df['queryObjId'].nunique()
        print(f"目标对象数: {unique_objs}")
    
    if 'difficulty' in df.columns:
        print('难度分布:')
        print(df['difficulty'].value_counts().to_string().replace('\n', '\n  '))
    
    if 'ambiguious' in df.columns:
        print('有歧义分布:')
        print(df['ambiguious'].value_counts().to_string().replace('\n', '\n  '))
    
    if 'annotation' in df.columns:
        print(f'目标描述(annotation)样例: {df["annotation"].iloc[0]}')

    print('\n[图像字段分析]')
    if 'image' in df.columns:
        example_rows = get_example_rows(df, example_count)
        for index, (_, row) in enumerate(example_rows.iterrows(), start=1):
            im_bytes = row['image']['bytes']
            img = Image.open(io.BytesIO(im_bytes))
            sample_path = IMAGE_SAMPLE_DIR / f'sample_{index - 1}.png'
            img.save(sample_path)
            print(
                f'  示例 {index}: sceneId={row.get("sceneId")}, '
                f'queryObjId={row.get("queryObjId")}, 图片尺寸={img.size}, 保存到 {sample_path}'
            )

    return df


def get_example_rows(df, example_count=3):
    """优先选不同 sceneId 的样本作为示例。"""
    if df is None or df.empty:
        return pd.DataFrame()
    if 'sceneId' in df.columns:
        return df.drop_duplicates('sceneId').head(example_count)
    return df.head(example_count)


def iter_npz_sources():
    """递归查找 data 目录下的 .npz 文件，也读取 zip 包里的 .npz 条目。"""
    seen_names = set()

    for path in sorted(DATA_DIR.rglob('*.npz')):
        seen_names.add(path.name)
        yield path.name, path, None

    for zip_path in sorted(DATA_DIR.rglob('*.zip')):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in sorted(zip_ref.namelist()):
                    if member.endswith('.npz') and Path(member).name not in seen_names:
                        yield Path(member).name, zip_path, member
        except Exception as e:
            print(f'读取 zip 失败: {zip_path} ({e})')


def load_npz(source_path, zip_member=None):
    """从普通 .npz 文件或 zip 内部 .npz 条目加载数据。"""
    if zip_member is None:
        return np.load(source_path, allow_pickle=True)

    with zipfile.ZipFile(source_path, 'r') as zip_ref:
        with zip_ref.open(zip_member) as f:
            return np.load(io.BytesIO(f.read()), allow_pickle=True)


def select_npz_examples(npz_sources, df=None, example_count=3):
    """选择与 parquet 示例 sceneId 对应的 npz；没有 parquet 时取前几个。"""
    if df is None or df.empty or 'sceneId' not in df.columns:
        return npz_sources[:example_count]

    source_by_scene = {Path(name).stem: (name, source_path, zip_member) for name, source_path, zip_member in npz_sources}
    selected = []
    for scene_id in get_example_rows(df, example_count)['sceneId']:
        source = source_by_scene.get(str(scene_id))
        if source:
            selected.append(source)
    return selected[:example_count]


def count_objects(npz):
    """估算场景中的对象实例数。"""
    if 'occlusion_objects' in npz and npz['occlusion_objects'].ndim >= 3:
        return npz['occlusion_objects'].shape[0]

    for mask_key in ('instances_objects.npy', 'instances_objects', 'instance_mask', 'mask'):
        if mask_key in npz:
            arr = npz[mask_key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                values = np.unique(arr)
                return int(np.count_nonzero(values))
            return arr.shape[0] if hasattr(arr, 'shape') and arr.ndim else len(arr)
    return None


def analyze_npz_files(df=None, example_count=3):
    """分析 data 目录下和 zip 包中的 npz 构成，只深度读取少量示例。"""
    print('\n' + '='*60)
    print('[3] 实例 mask npz 文件分析')
    print('='*60)

    npz_sources = list(iter_npz_sources())
    if not npz_sources:
        print(f'未在 {DATA_DIR} 下找到 .npz 文件或包含 .npz 的 zip 文件。')
        return {'total_npz': 0, 'missing_npz': 0}

    print(f'发现 {len(npz_sources)} 个 npz 数据源（含 zip 内文件）')
    print('前几个 npz 数据源:')
    for name, source_path, zip_member in npz_sources[:10]:
        if zip_member:
            print(f'  - {source_path.name}!/{zip_member}')
        else:
            print(f'  - {source_path.relative_to(DATA_DIR)} ({format_size(source_path.stat().st_size)})')
    
    npz_names = {Path(name).stem for name, _, _ in npz_sources}
    if df is not None and 'sceneId' in df.columns:
        scene_ids = {str(sid) for sid in df['sceneId'].dropna().unique()}
        missing_npz = len(scene_ids - npz_names)
        print(f'parquet 场景对应检查: {len(scene_ids) - missing_npz} 个存在，{missing_npz} 个缺失')
    else:
        missing_npz = 0

    example_sources = select_npz_examples(npz_sources, df, example_count)
    print(f'\n只读取 {len(example_sources)} 个 npz 示例做结构分析:')

    object_counts = []
    failed = 0

    for index, (name, source_path, zip_member) in enumerate(example_sources, start=1):
        try:
            with load_npz(source_path, zip_member) as npz:
                object_count = count_objects(npz)
                if object_count is not None:
                    object_counts.append(object_count)

                print(f'\n  [npz 示例 {index}] {name}')
                print(f'    对象实例数估计: {object_count}')
                for key in npz.files:
                    arr = npz[key]
                    print(f'    - {key}: {describe_array(arr)}')
        except Exception as e:
            failed += 1
            print(f'读取 {name} 失败: {e}')

    print(f'\nnpz 示例读取结果: {len(example_sources) - failed} 个成功，{failed} 个失败')
    if object_counts:
        print(f'3 个示例对象实例数: {object_counts}')

    return {'total_npz': len(npz_sources), 'missing_npz': missing_npz}


def analyze_examples(df, example_count=3):
    """把 parquet 标注、图片和 npz 按 sceneId 对齐展示。"""
    if df is None or df.empty:
        return

    print('\n' + '='*60)
    print(f'[4] {example_count} 个样本级示例')
    print('='*60)

    example_rows = get_example_rows(df, example_count)
    for index, (_, row) in enumerate(example_rows.iterrows(), start=1):
        print(f'\n  [样本示例 {index}]')
        print(f'    sceneId: {row.get("sceneId")}')
        print(f'    queryObjId: {row.get("queryObjId")}')
        print(f'    annotation: {row.get("annotation")}')
        print(f'    groundTruthObjIds: {row.get("groundTruthObjIds")}')
        print(f'    difficulty: {row.get("difficulty")}')
        print(f'    ambiguious: {row.get("ambiguious")}')
        print(f'    split: {row.get("split")}')


def visualize_scene_npz(scene_id):
    """可视化指定 sceneId 的 npz 内容并保存图片。"""
    print('\n' + '='*60)
    print(f'[5] 指定场景可视化: sceneId={scene_id}')
    print('='*60)

    selected_source = None
    selected_name = f'{scene_id}.npz'

    for name, source_path, zip_member in iter_npz_sources():
        if Path(name).stem == str(scene_id):
            selected_source = (name, source_path, zip_member)
            break

    if selected_source is None:
        print(f'未找到 sceneId={scene_id} 对应的 npz 文件。')
        return

    name, source_path, zip_member = selected_source
    out_dir = NPZ_VIZ_DIR / f'scene_{scene_id}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if zip_member:
        source_desc = f'{source_path}!/{zip_member}'
    else:
        source_desc = str(source_path)

    print(f'数据源: {source_desc}')

    saved_files = []
    with load_npz(source_path, zip_member) as npz:
        for key in npz.files:
            arr = npz[key]
            if not isinstance(arr, np.ndarray):
                continue

            if arr.ndim == 2:
                fig, ax = plt.subplots(figsize=(6, 6))
                cmap = 'viridis' if np.issubdtype(arr.dtype, np.number) else 'gray'
                artist = ax.imshow(arr, cmap=cmap)
                ax.set_title(f'scene {scene_id} - {key}')
                ax.axis('off')
                if np.issubdtype(arr.dtype, np.number):
                    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
                out_path = out_dir / f'{key}.png'
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(out_path)

            elif arr.ndim == 3:
                max_slices = min(4, arr.shape[0])
                fig, axes = plt.subplots(1, max_slices, figsize=(4 * max_slices, 4))
                if max_slices == 1:
                    axes = [axes]

                for index in range(max_slices):
                    slice_arr = arr[index]
                    cmap = 'magma' if np.issubdtype(slice_arr.dtype, np.number) else 'gray'
                    artist = axes[index].imshow(slice_arr, cmap=cmap)
                    axes[index].set_title(f'{key}[{index}]')
                    axes[index].axis('off')
                    if np.issubdtype(slice_arr.dtype, np.number):
                        fig.colorbar(artist, ax=axes[index], fraction=0.046, pad=0.04)

                fig.suptitle(f'scene {scene_id} - {key} (first {max_slices} slices)')
                out_path = out_dir / f'{key}_slices.png'
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(out_path)

    print(f'可视化输出目录: {out_dir}')
    if saved_files:
        print('已保存文件:')
        for file_path in saved_files:
            print(f'  - {file_path}')
    else:
        print(f'{selected_name} 未发现可视化数组。')


def build_occlusion_digraph_from_npz(npz, threshold=0.01):
    """从 npz 的 occlusion_objects 和 instances_objects 构建遮挡有向图。"""
    if 'occlusion_objects' not in npz or 'instances_objects' not in npz:
        raise ValueError('npz 中缺少 occlusion_objects 或 instances_objects，无法构图。')

    occlusion_objects = np.asarray(npz['occlusion_objects'])
    instances_objects = np.asarray(npz['instances_objects'])

    if occlusion_objects.ndim != 3:
        raise ValueError(f'occlusion_objects 维度应为3，当前为 {occlusion_objects.ndim}')
    if instances_objects.ndim != 2:
        raise ValueError(f'instances_objects 维度应为2，当前为 {instances_objects.ndim}')

    num_objects = occlusion_objects.shape[0]
    graph = nx.DiGraph()

    object_pixel_counts = {}
    for obj_id in range(1, num_objects + 1):
        obj_mask = (instances_objects == obj_id)
        pixel_count = int(np.count_nonzero(obj_mask))
        if pixel_count > 0:
            object_pixel_counts[obj_id] = pixel_count
            graph.add_node(obj_id, pixel_count=pixel_count)

    edge_stats = []
    for source_id, source_pixels in object_pixel_counts.items():
        occ_map = occlusion_objects[source_id - 1]
        for target_id in object_pixel_counts.keys():
            if source_id == target_id:
                continue
            overlap = np.logical_and(occ_map == target_id, instances_objects == source_id)
            overlap_pixels = int(np.count_nonzero(overlap))
            if overlap_pixels == 0:
                continue

            ratio = overlap_pixels / source_pixels
            if ratio >= threshold:
                graph.add_edge(target_id, source_id, ratio=ratio, overlap_pixels=overlap_pixels)
                edge_stats.append((target_id, source_id, ratio, overlap_pixels))

    return graph, edge_stats


def visualize_scene_occlusion_graph(scene_id, threshold=0.01):
    """为指定 sceneId 生成遮挡关系有向图可视化。"""
    print('\n' + '=' * 60)
    print(f'[6] 遮挡有向图可视化: sceneId={scene_id}, threshold={threshold:.2%}')
    print('=' * 60)

    selected_source = None
    for name, source_path, zip_member in iter_npz_sources():
        if Path(name).stem == str(scene_id):
            selected_source = (name, source_path, zip_member)
            break

    if selected_source is None:
        print(f'未找到 sceneId={scene_id} 对应的 npz 文件。')
        return

    _, source_path, zip_member = selected_source
    out_dir = NPZ_VIZ_DIR / f'scene_{scene_id}'
    out_dir.mkdir(parents=True, exist_ok=True)

    with load_npz(source_path, zip_member) as npz:
        graph, edge_stats = build_occlusion_digraph_from_npz(npz, threshold=threshold)

    print(f'节点数: {graph.number_of_nodes()}，边数: {graph.number_of_edges()}')
    if edge_stats:
        print('前几条边 (遮挡者 -> 被遮挡者, 比例):')
        for src, dst, ratio, pixels in sorted(edge_stats, key=lambda x: x[2], reverse=True)[:10]:
            print(f'  - {src} -> {dst}, ratio={ratio:.3%}, pixels={pixels}')

    if graph.number_of_nodes() == 0:
        print('没有可用节点，跳过绘图。')
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)

    node_sizes = [max(400, min(3000, graph.nodes[node].get('pixel_count', 1) / 20)) for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='#8ecae6', ax=ax)
    nx.draw_networkx_labels(graph, pos, labels={node: str(node) for node in graph.nodes()}, font_size=10, ax=ax)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle='-|>', arrowsize=18, width=1.8, edge_color='#219ebc', ax=ax)

    edge_labels = {(u, v): f"{d['ratio']:.1%}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    ax.set_title(f'Scene {scene_id} Occlusion Graph (threshold={threshold:.1%})')
    ax.axis('off')

    out_path = out_dir / f'occlusion_graph_threshold_{int(threshold * 1000):03d}.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存遮挡有向图: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='SmartGrasp 数据分析工具')
    parser.add_argument('--scene-id', type=int, default=None, help='可选：指定 sceneId，并可视化对应 npz')
    parser.add_argument('--example-count', type=int, default=3, help='分析时展示的示例数量')
    parser.add_argument('--occlusion-threshold', type=float, default=0.01, help='遮挡边保留阈值（默认 0.01 即 1%）')
    args = parser.parse_args()

    example_count = args.example_count
    # 分析目录结构
    analyze_data_directory()

    df = analyze_parquet_files(example_count)
    npz_result = analyze_npz_files(df, example_count)
    analyze_examples(df, example_count)

    if args.scene_id is not None:
        visualize_scene_npz(args.scene_id)
        visualize_scene_occlusion_graph(args.scene_id, threshold=args.occlusion_threshold)

    print('\n' + '='*60)
    print('分析完成！')
    print('='*60)
    if df is not None and 'sceneId' in df.columns:
        unique_scenes = df['sceneId'].nunique()
        print(f'总结: {len(df)} 个样本，{unique_scenes} 个场景，{npz_result["total_npz"]} 个 npz 文件可用')
    else:
        print(f'总结: {npz_result["total_npz"]} 个 npz 文件可用')

if __name__ == "__main__":
    main()
