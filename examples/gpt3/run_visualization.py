import os
import sys
import json
import argparse
from datetime import datetime

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("NetworkX가 설치되어 있지 않습니다. 'pip install networkx'로 설치해주세요.")

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib이 설치되어 있지 않습니다. 'pip install matplotlib'로 설치해주세요.")

from syntorch.core.trace import trace_manager


def load_json_graph(json_file):
    """JSON 파일에서 그래프 로드"""
    if not HAS_NETWORKX:
        print("NetworkX가 설치되어 있지 않아 그래프를 로드할 수 없습니다.")
        return None

    try:
        with open(json_file, "r") as f:
            graph_data = json.load(f)

        G = nx.node_link_graph(graph_data)
        print(f"그래프 로드 완료: 노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개")
        return G
    except Exception as e:
        print(f"그래프 로드 중 오류 발생: {e}")
        return None


def visualize_graph(G, output_file=None, show=True):
    """그래프 시각화"""
    if not HAS_NETWORKX or not HAS_MATPLOTLIB:
        print("NetworkX 또는 Matplotlib이 설치되어 있지 않아 시각화할 수 없습니다.")
        return

    if G is None or G.number_of_nodes() == 0:
        print("시각화할 그래프가 없습니다.")
        return

    plt.figure(figsize=(16, 12))

    # 노드 색상 및 레이블 설정
    node_colors = []
    node_labels = {}

    for node in G.nodes():
        node_type = G.nodes[node].get("type", "")
        if node_type == "operation":
            node_colors.append("lightblue")
            node_labels[node] = G.nodes[node].get("op_name", str(node)[:10])
        elif node_type == "tensor":
            node_colors.append("lightgreen")
            node_labels[node] = str(node)[:10]  # ID 값을 짧게 표시
        else:
            node_colors.append("gray")
            node_labels[node] = str(node)[:10]

    # 레이아웃 설정 (large_graph=True는 큰 그래프에 적합)
    # pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)  # scipy 필요
    pos = nx.circular_layout(G)  # scipy 필요 없음

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True, arrowsize=10)

    # 레이블 그리기
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    plt.title("Tensor Operation Graph", fontsize=16)
    plt.axis("off")

    # 파일로 저장
    if output_file:
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        print(f"그래프가 {output_file}에 저장되었습니다.")

    # 화면에 표시
    if show:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="텐서 트레이스 그래프 시각화")
    parser.add_argument(
        "--input", type=str, default="tensor_trace_graph.json", help="입력 JSON 그래프 파일"
    )
    parser.add_argument(
        "--output", type=str, default="tensor_trace_visualization.png", help="출력 이미지 파일"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "svg", "chakra", "json", "graphml"],
        help="출력 파일 형식",
    )
    parser.add_argument("--show", action="store_true", help="시각화 결과 화면에 표시")
    args = parser.parse_args()

    # 출력 파일 이름 설정
    output_file = args.output
    if args.format == "svg" and not output_file.endswith(".svg"):
        output_file = output_file.rsplit(".", 1)[0] + ".svg"
    elif args.format == "chakra" and not output_file.endswith(".json"):
        output_file = output_file.rsplit(".", 1)[0] + ".json"
    elif args.format == "graphml" and not output_file.endswith(".graphml"):
        output_file = output_file.rsplit(".", 1)[0] + ".graphml"

    # JSON 파일에서 그래프 로드
    G = load_json_graph(args.input)
    if G:
        # 출력 형식에 따라 처리
        if args.format == "png":
            # 그래프 시각화 (PNG)
            visualize_graph(G, output_file, args.show)
        elif args.format == "svg":
            # SVG로 내보내기
            export_graph_svg(G, output_file)
            print(f"그래프가 SVG 형식으로 {output_file}에 저장되었습니다.")
        elif args.format == "chakra":
            # Chakra 형식으로 내보내기
            export_graph_chakra(G, output_file)
            print(f"그래프가 MLCommons Chakra 형식으로 {output_file}에 저장되었습니다.")
        elif args.format == "graphml":
            # GraphML 형식으로 내보내기
            nx.write_graphml(G, output_file)
            print(f"그래프가 GraphML 형식으로 {output_file}에 저장되었습니다.")

        print(f"그래프의 기본 통계:")
        print(f"- 노드 수: {G.number_of_nodes()}")
        print(f"- 엣지 수: {G.number_of_edges()}")

        # 노드 유형별 개수
        node_types = {}
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print("\n노드 유형별 개수:")
        for node_type, count in node_types.items():
            print(f"- {node_type}: {count}개")

        # 연산 유형별 개수 (operation 노드만)
        if "operation" in node_types:
            op_types = {}
            for node in G.nodes():
                if G.nodes[node].get("type") == "operation":
                    op_name = G.nodes[node].get("op_name", "unknown")
                    op_types[op_name] = op_types.get(op_name, 0) + 1

            print("\n연산 유형별 개수:")
            for op_name, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
                print(f"- {op_name}: {count}개")


def export_graph_svg(G, filepath):
    """그래프를 SVG 형식으로 내보내기"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib이 설치되어 있지 않아 SVG로 내보낼 수 없습니다.")
        return

    plt.figure(figsize=(16, 12))

    # 노드 색상 및 레이블 설정
    node_colors = []
    node_labels = {}

    for node in G.nodes():
        node_type = G.nodes[node].get("type", "")
        if node_type == "operation":
            node_colors.append("lightblue")
            op_name = G.nodes[node].get("op_name", "")
            node_labels[node] = op_name if op_name else str(node)[:10]
        elif node_type == "tensor":
            node_colors.append("lightgreen")
            shape = G.nodes[node].get("shape", None)
            dtype = G.nodes[node].get("dtype", None)
            label = ""
            if shape is not None:
                if isinstance(shape, (tuple, list)):
                    shape_str = "x".join(str(s) for s in shape)
                    label += f"{shape_str}"
                else:
                    label += f"{shape}"
            if dtype is not None:
                label += f"\n{dtype}"
            if not label:
                label = str(node)[:10]
            node_labels[node] = label
        else:
            node_colors.append("gray")
            node_labels[node] = str(node)[:10]

    # 레이아웃 설정
    pos = nx.circular_layout(G)

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8)

    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True, arrowsize=15)

    # 레이블 그리기
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    plt.title("텐서 연산 그래프")
    plt.axis("off")

    # 그래프 정보 표시
    plt.figtext(
        0.02,
        0.02,
        f"노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}",
        fontsize=9,
        ha="left",
    )

    # 범례 추가
    import matplotlib.patches as mpatches

    tensor_patch = mpatches.Patch(color="lightgreen", label="텐서 (shape, dtype)")
    op_patch = mpatches.Patch(color="lightblue", label="연산")
    plt.legend(handles=[tensor_patch, op_patch], loc="upper right")

    # SVG로 저장
    plt.savefig(filepath, format="svg", bbox_inches="tight", dpi=300)
    plt.close()


def export_graph_chakra(G, filepath):
    """그래프를 MLCommons Chakra 형식으로 내보내기"""
    # Chakra 형식 기본 구조
    chakra = {
        "schema_version": "0.1.0",
        "name": "syntorch_model",
        "metadata": {
            "framework": "syntorch",
            "description": "Syntorch Tensor Operation Graph",
            "created_at": datetime.now().isoformat(),
        },
        "graph": {"nodes": [], "edges": []},
    }

    # 노드 ID 맵핑 (NetworkX 노드 ID -> Chakra 노드 ID)
    node_id_map = {}

    # 노드 변환
    for i, node in enumerate(G.nodes()):
        node_data = G.nodes[node]
        node_type = node_data.get("type", "unknown")

        # Chakra 노드 ID 생성
        chakra_node_id = f"n{i}"
        node_id_map[node] = chakra_node_id

        # 노드 메타데이터 설정
        attributes = {}
        if node_type == "tensor":
            shape = node_data.get("shape", None)
            dtype = node_data.get("dtype", None)
            if shape is not None:
                attributes["shape"] = shape
            if dtype is not None:
                attributes["dtype"] = dtype
        elif node_type == "operation":
            op_name = node_data.get("op_name", "unknown")
            attributes["operation"] = op_name
            # 연산 속성 추가
            for key, value in node_data.items():
                if key not in ["type", "op_name"]:
                    attributes[key] = value

        # Chakra 노드 생성
        chakra_node = {"id": chakra_node_id, "type": node_type, "attributes": attributes}
        chakra["graph"]["nodes"].append(chakra_node)

    # 엣지 변환
    for i, (src, dst) in enumerate(G.edges()):
        edge_data = G.get_edge_data(src, dst)

        # Chakra 엣지 생성
        chakra_edge = {
            "id": f"e{i}",
            "source": node_id_map[src],
            "target": node_id_map[dst],
            "attributes": edge_data if edge_data else {},
        }
        chakra["graph"]["edges"].append(chakra_edge)

    # JSON으로 저장
    with open(filepath, "w") as f:
        json.dump(chakra, f, indent=2)


if __name__ == "__main__":
    main()
