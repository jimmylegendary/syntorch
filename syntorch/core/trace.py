import functools
from typing import Callable, Any, Dict, List, Tuple
from abc import ABC, ABCMeta
import numpy as np

# NetworkX를 임포트 시도하고 사용 가능한지 확인
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("NetworkX가 설치되어 있지 않습니다. 'pip install networkx'로 설치해주세요.")

# 플롯팅 라이브러리를 임포트 시도
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib이 설치되어 있지 않습니다. 'pip install matplotlib'로 설치해주세요.")


class TraceManager:
    def __init__(self):
        self.logs = []
        self.graph = {}  # 그래프 구조로 트레이스 기록
        self.nx_graph = None  # NetworkX 그래프

    def record_call(self, cls_name: str, method_name: str, args, kwargs, result):
        """기본 메서드 호출 기록"""
        self.logs.append((cls_name, method_name, args, kwargs, result))

    def trace_tensor(self, op_name: str, input_tensors: List, output_tensor, metadata: Dict = None):
        """텐서 연산을 그래프 구조로 기록

        Args:
            op_name: 연산 이름 (matmul, view, transpose 등)
            input_tensors: 입력 텐서 목록
            output_tensor: 출력 텐서
            metadata: 추가 메타데이터
        """
        # 메타데이터가 없으면 빈 딕셔너리 생성
        if metadata is None:
            metadata = {}

        # 연산 노드의 고유 ID 생성
        op_node_id = f"op_{op_name}_{id(output_tensor)}"

        # 텐서가 아닌 입력값들을 저장할 딕셔너리
        non_tensor_inputs = {}

        # 각 입력 텐서에 대해 엣지 생성
        tensor_inputs = []
        for i, input_obj in enumerate(input_tensors):
            if hasattr(input_obj, "_id") and hasattr(input_obj, "data"):  # Tensor 객체인 경우
                input_id = input_obj._id
                tensor_inputs.append(input_id)

                # 텐서 노드에 shape, dtype 정보 추가
                tensor_metadata = {
                    "type": "tensor",
                    "shape": input_obj.shape if hasattr(input_obj, "shape") else None,
                    "dtype": input_obj.dtype if hasattr(input_obj, "dtype") else None,
                }

                # 입력 텐서 -> 연산 엣지 저장
                self.graph.setdefault(input_id, []).append(
                    {
                        "op": op_name,
                        "op_node": op_node_id,
                        "output": (
                            id(output_tensor) if hasattr(output_tensor, "_id") else output_tensor
                        ),
                        "metadata": tensor_metadata,
                    }
                )
            else:
                # 텐서가 아닌 입력은 연산 노드의 속성으로 저장
                non_tensor_inputs[f"param_{i}"] = str(input_obj)

        # 출력 텐서 노드 정보
        if hasattr(output_tensor, "_id") and hasattr(output_tensor, "data"):
            output_id = output_tensor._id
            output_metadata = {
                "type": "tensor",
                "shape": output_tensor.shape if hasattr(output_tensor, "shape") else None,
                "dtype": output_tensor.dtype if hasattr(output_tensor, "dtype") else None,
            }
        else:
            output_id = id(output_tensor)
            output_metadata = {"type": "tensor"}

        # 연산 노드에 비텐서 입력값 정보 포함
        metadata.update(non_tensor_inputs)

        # 그래프에 연산 노드 정보 추가 (연산 노드 -> 출력 텐서 엣지 포함)
        self.graph.setdefault(op_node_id, []).append(
            {
                "op": op_name,
                "inputs": tensor_inputs,
                "output": output_id,
                "metadata": {**metadata, "type": "operation"},
            }
        )

    def trace_memory(self, address: int, size: int, operation: str, data=None):
        """메모리 연산 트레이스"""
        self.logs.append(("Memory", operation, {"address": address, "size": size, "data": data}))

    def trace_hw(self, component_id: str, operation: str, duration: float, args=None):
        """하드웨어 연산 트레이스"""
        self.logs.append(
            ("HW", operation, {"component": component_id, "duration": duration, "args": args})
        )

    def trace_comm_op(
        self, op_name: str, input_tensors: List, output_tensor, metadata: Dict = None
    ):
        """집합 통신 연산을 그래프 구조로 기록

        Args:
            op_name: 연산 이름 (all_reduce, all_gather, reduce_scatter, broadcast 등)
            input_tensors: 입력 텐서 목록
            output_tensor: 출력 텐서
            metadata: 추가 메타데이터 (group_size, group_rank, comm_type 등)
        """
        # 메타데이터가 없으면 빈 딕셔너리 생성
        if metadata is None:
            metadata = {}

        # 연산 노드의 고유 ID 생성 (통신 연산은 'comm_'으로 시작하도록 함)
        op_node_id = f"comm_{op_name}_{id(output_tensor)}"

        # 각 입력 텐서에 대해 엣지 생성
        tensor_inputs = []
        for i, input_obj in enumerate(input_tensors):
            if hasattr(input_obj, "_id") and hasattr(input_obj, "data"):  # Tensor 객체인 경우
                input_id = input_obj._id
                tensor_inputs.append(input_id)

                # 텐서 노드에 shape, dtype 정보 추가
                tensor_metadata = {
                    "type": "tensor",
                    "shape": input_obj.shape if hasattr(input_obj, "shape") else None,
                    "dtype": input_obj.dtype if hasattr(input_obj, "dtype") else None,
                }

                # 입력 텐서 -> 연산 엣지 저장
                self.graph.setdefault(input_id, []).append(
                    {
                        "op": op_name,
                        "op_node": op_node_id,
                        "output": (
                            id(output_tensor) if hasattr(output_tensor, "_id") else output_tensor
                        ),
                        "metadata": tensor_metadata,
                    }
                )

        # 출력 텐서 노드 정보
        if hasattr(output_tensor, "_id") and hasattr(output_tensor, "data"):
            output_id = output_tensor._id
            output_metadata = {
                "type": "tensor",
                "shape": output_tensor.shape if hasattr(output_tensor, "shape") else None,
                "dtype": output_tensor.dtype if hasattr(output_tensor, "dtype") else None,
            }
        else:
            output_id = id(output_tensor)
            output_metadata = {"type": "tensor"}

        # 통신 특정 메타데이터 추가
        comm_metadata = {
            "type": "comm_op",
            "comm_op": op_name,
            "group_size": metadata.get("group_size", 1),
            "group_rank": metadata.get("group_rank", 0),
            "comm_type": metadata.get("comm_type", op_name),
        }

        # 그래프에 연산 노드 정보 추가 (연산 노드 -> 출력 텐서 엣지 포함)
        self.graph.setdefault(op_node_id, []).append(
            {
                "op": op_name,
                "inputs": tensor_inputs,
                "output": output_id,
                "metadata": {**metadata, **comm_metadata},
            }
        )

    def visualize_comm_graph(self, filename=None, show=True):
        """통신 연산이 강조된 그래프 시각화"""
        if not HAS_NETWORKX or not HAS_MATPLOTLIB:
            print("NetworkX 또는 Matplotlib이 설치되어 있지 않습니다.")
            return

        G = self.get_networkx_graph()
        if G is None or G.number_of_nodes() == 0:
            print("시각화할 그래프가 없습니다.")
            return

        plt.figure(figsize=(16, 12))

        # 노드 색상 및 레이블 설정
        node_colors = []
        node_labels = {}
        node_shapes = []
        node_sizes = []

        for node in G.nodes():
            node_type = G.nodes[node].get("type", "")

            if node_type == "comm_op":
                # 통신 연산 노드는 빨간색으로 표시
                node_colors.append("red")
                op_name = G.nodes[node].get("comm_op", "")
                node_labels[node] = op_name if op_name else str(node)[:10]
                node_shapes.append("s")  # 사각형
                node_sizes.append(1000)  # 더 크게
            elif node_type == "operation":
                node_colors.append("lightblue")
                op_name = G.nodes[node].get("op_name", "")
                node_labels[node] = op_name if op_name else str(node)[:10]
                node_shapes.append("o")  # 원형
                node_sizes.append(800)
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
                node_shapes.append("o")  # 원형
                node_sizes.append(800)
            else:
                node_colors.append("gray")
                node_labels[node] = str(node)[:10]
                node_shapes.append("o")  # 원형
                node_sizes.append(800)

        # 레이아웃 설정
        pos = nx.circular_layout(G)  # scipy 없이도 작동

        # 노드 그리기
        for i, (node, pos_node) in enumerate(pos.items()):
            nx.draw_networkx_nodes(
                G,
                {node: pos_node},
                node_color=[node_colors[i]],
                node_size=node_sizes[i],
                alpha=0.8,
                node_shape=node_shapes[i],
            )

        # 엣지 그리기
        # 통신 연산 관련 엣지는 빨간색으로 표시
        normal_edges = []
        comm_edges = []

        for u, v in G.edges():
            if G.nodes[u].get("type") == "comm_op" or G.nodes[v].get("type") == "comm_op":
                comm_edges.append((u, v))
            else:
                normal_edges.append((u, v))

        # 일반 엣지 그리기
        nx.draw_networkx_edges(
            G, pos, edgelist=normal_edges, width=1.0, alpha=0.5, arrows=True, arrowsize=15
        )

        # 통신 엣지 그리기
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=comm_edges,
            width=2.0,
            alpha=0.7,
            arrows=True,
            arrowsize=20,
            edge_color="red",
        )

        # 레이블 그리기
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

        plt.title("Tensor 연산 및 통신 그래프")
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

        tensor_patch = mpatches.Patch(color="lightgreen", label="텐서")
        op_patch = mpatches.Patch(color="lightblue", label="연산")
        comm_patch = mpatches.Patch(color="red", label="통신 연산")
        plt.legend(handles=[tensor_patch, op_patch, comm_patch], loc="upper right")

        # 파일로 저장
        if filename:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
            print(f"그래프가 {filename}에 저장되었습니다.")

        # 화면에 표시
        if show:
            plt.show()

        plt.close()

    def _build_networkx_graph(self):
        """내부 그래프 구조를 NetworkX 그래프로 변환"""
        if not HAS_NETWORKX:
            print("NetworkX가 설치되어 있지 않습니다.")
            return None

        G = nx.DiGraph()  # 방향 그래프 생성

        # 노드와 엣지 추가
        for source_id, outputs in self.graph.items():
            # 소스 노드가 연산자인지 텐서인지 구분
            is_op_node = isinstance(source_id, str) and (
                source_id.startswith("op_") or source_id.startswith("comm_")
            )

            for output_info in outputs:
                op_name = output_info["op"]
                output_id = output_info["output"]
                metadata = output_info["metadata"]

                # 노드 유형 판단
                node_type = "operation"
                if isinstance(source_id, str):
                    if source_id.startswith("comm_"):
                        node_type = "comm_op"

                if is_op_node:
                    # 소스가 연산 노드인 경우
                    # 소스 노드 추가
                    if not G.has_node(source_id):
                        if node_type == "comm_op":
                            # 통신 연산 노드의 경우 통신 관련 메타데이터 추가
                            comm_metadata = {"type": "comm_op", "comm_op": op_name, **metadata}
                            G.add_node(source_id, **comm_metadata)
                        else:
                            G.add_node(source_id, type="operation", op_name=op_name, **metadata)

                    # 출력 텐서 노드 추가
                    if not G.has_node(output_id):
                        G.add_node(output_id, **metadata)

                    # 연산 -> 출력 텐서 엣지 추가
                    G.add_edge(source_id, output_id)

                    # 입력 텐서 -> 연산 엣지 추가 (있는 경우)
                    if "inputs" in output_info:
                        for input_id in output_info["inputs"]:
                            if not G.has_node(input_id):
                                G.add_node(input_id, type="tensor")
                            G.add_edge(input_id, source_id)
                else:
                    # 소스가 텐서 노드인 경우
                    # 텐서 노드 추가
                    tensor_props = {}
                    if "shape" in metadata:
                        tensor_props["shape"] = metadata["shape"]
                    if "dtype" in metadata:
                        tensor_props["dtype"] = metadata["dtype"]

                    if not G.has_node(source_id):
                        G.add_node(source_id, type="tensor", **tensor_props)

                    # 연산 노드의 ID
                    op_node_id = output_info.get(
                        "op_node", f"op_{op_name}_{hash((source_id, output_id))}"
                    )

                    # 연산 노드 추가
                    if not G.has_node(op_node_id):
                        if op_node_id.startswith("comm_"):
                            # 통신 연산 노드인 경우
                            G.add_node(op_node_id, type="comm_op", comm_op=op_name)
                        else:
                            G.add_node(op_node_id, type="operation", op_name=op_name)

                    # 출력 텐서 노드 추가
                    if not G.has_node(output_id):
                        G.add_node(output_id, type="tensor")

                    # 텐서 -> 연산 -> 출력 텐서 엣지 추가
                    G.add_edge(source_id, op_node_id)
                    G.add_edge(op_node_id, output_id)

        return G

    def get_networkx_graph(self):
        """NetworkX 그래프 반환 (아직 생성되지 않았다면 생성)"""
        if self.nx_graph is None:
            self.nx_graph = self._build_networkx_graph()
        return self.nx_graph

    def visualize_graph(self, filename=None, show=True):
        """그래프 시각화"""
        if not HAS_NETWORKX or not HAS_MATPLOTLIB:
            print("NetworkX 또는 Matplotlib이 설치되어 있지 않습니다.")
            return

        G = self.get_networkx_graph()
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
                op_name = G.nodes[node].get("op_name", "")
                # 연산 노드 레이블에 연산 이름 표시
                node_labels[node] = op_name if op_name else str(node)[:10]
            elif node_type == "tensor":
                node_colors.append("lightgreen")
                # 텐서 노드 레이블에 shape과 dtype 표시
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
        pos = nx.circular_layout(G)  # scipy 없이도 작동

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

        # 파일로 저장
        if filename:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
            print(f"그래프가 {filename}에 저장되었습니다.")

        # 화면에 표시
        if show:
            plt.show()

        plt.close()

    def export_graph(self, format="graphml", filepath=None):
        """그래프를 다양한 형식으로 내보내기"""
        if not HAS_NETWORKX:
            print("NetworkX가 설치되어 있지 않습니다.")
            return

        G = self.get_networkx_graph()
        if G is None:
            print("내보낼 그래프가 없습니다.")
            return

        if filepath is None:
            filepath = f"tensor_graph.{format}"

        if format == "graphml":
            nx.write_graphml(G, filepath)
        elif format == "gexf":
            nx.write_gexf(G, filepath)
        elif format == "adjlist":
            nx.write_adjlist(G, filepath)
        elif format == "edgelist":
            nx.write_edgelist(G, filepath)
        elif format == "json":
            import json

            # NetworkX 그래프를 사전 형태로 변환
            data = nx.node_link_data(G)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "svg":
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
                    # 텐서 노드 레이블에 shape과 dtype 표시
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
            pos = nx.circular_layout(G)  # scipy 없이도 작동

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
            plt.savefig(filepath, format="svg", bbox_inches="tight")
            plt.close()

        elif format == "chakra":
            # MLCommons Chakra 포맷으로 변환
            chakra_data = self._convert_to_chakra(G)
            import json

            with open(filepath, "w") as f:
                json.dump(chakra_data, f, indent=2)
        else:
            print(f"지원되지 않는 형식: {format}")
            return

        print(f"그래프가 {filepath}에 {format} 형식으로 저장되었습니다.")

    def _convert_to_chakra(self, G):
        """NetworkX 그래프를 MLCommons Chakra 포맷으로 변환"""
        chakra = {
            "schema_version": "0.1.0",
            "name": "syntorch_model",
            "metadata": {
                "framework": "syntorch",
                "description": "Syntorch Tensor Operation Graph",
                "created_at": self._get_current_timestamp(),
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

        return chakra

    def _get_current_timestamp(self):
        """현재 시간을 ISO 8601 형식으로 반환"""
        from datetime import datetime

        return datetime.now().isoformat()

    def dump(self, format="json", filepath=None):
        """트레이스 로그 출력"""
        if format == "console":
            for entry in self.logs:
                print("[Trace]", entry)
        elif format == "graph" and HAS_NETWORKX:
            self.visualize_graph(filename=filepath)
        # JSON 출력 등 구현...


# 전역 싱글톤
trace_manager = TraceManager()


def auto_trace(method: Callable):
    """메타클래스에서 메서드를 감쌀 때 사용하는 내부용 데코레이터"""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        cls_name = self.__class__.__name__
        method_name = method.__name__
        trace_manager.record_call(cls_name, method_name, args, kwargs, result)
        return result

    return wrapper


class AutoTraceMeta(ABCMeta):
    """
    이 메타클래스를 상속받으면,
    해당 클래스의 모든 'callable 속성' (즉, 메서드)에
    auto_trace 데코레이터가 자동으로 적용됨
    """

    def __new__(mcs, name, bases, namespace):
        new_namespace = {}
        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and not attr_name.startswith("__"):
                # 마법: 메서드를 auto_trace로 감싼다
                new_namespace[attr_name] = auto_trace(attr_value)
            else:
                new_namespace[attr_name] = attr_value
        return super().__new__(mcs, name, bases, new_namespace)


class SyntorchLayer(ABC, metaclass=AutoTraceMeta):
    """모든 Syntorch 레이어의 기본 클래스"""

    pass
