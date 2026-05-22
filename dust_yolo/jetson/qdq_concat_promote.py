import argparse
from collections import defaultdict
import onnx
from onnx import helper


DEFAULT_RULES = {
    "/model.20/Concat": "/model.21/cv1/conv/_input_quantizer/QuantizeLinear",
    "/model.23/Concat": "/model.24/cv1/conv/_input_quantizer/QuantizeLinear",
    "/model.26/Concat": "/model.27/cv1/conv/_input_quantizer/QuantizeLinear",
    "/model.21/Concat": "/model.21/cv3/conv/_input_quantizer/QuantizeLinear",
    "/model.24/Concat": "/model.24/cv3/conv/_input_quantizer/QuantizeLinear",
    "/model.27/Concat": "/model.27/cv3/conv/_input_quantizer/QuantizeLinear",
    "/model.30/Concat": "/model.30/cv3/conv/_input_quantizer/QuantizeLinear",
}


UNIFORM_ADD_SCALE_MAP = {
    "/model.2/m/m.0/addop/Add": "/model.2/m/m.0/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.2/m/m.1/addop/Add": "/model.2/m/m.1/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.2/m/m.2/addop/Add": "/model.2/m/m.2/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.4/m/m.0/addop/Add": "/model.4/m/m.0/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.4/m/m.1/addop/Add": "/model.4/m/m.1/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.4/m/m.2/addop/Add": "/model.4/m/m.2/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.4/m/m.3/addop/Add": "/model.4/m/m.3/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.4/m/m.4/addop/Add": "/model.4/m/m.4/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.4/m/m.5/addop/Add": "/model.4/m/m.5/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.6/m/m.0/addop/Add": "/model.6/m/m.0/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.6/m/m.1/addop/Add": "/model.6/m/m.1/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.6/m/m.2/addop/Add": "/model.6/m/m.2/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.6/m/m.3/addop/Add": "/model.6/m/m.3/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.6/m/m.4/addop/Add": "/model.6/m/m.4/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.6/m/m.5/addop/Add": "/model.6/m/m.5/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.6/m/m.6/addop/Add": "/model.6/m/m.6/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.6/m/m.7/addop/Add": "/model.6/m/m.7/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.6/m/m.8/addop/Add": "/model.6/m/m.8/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.8/m/m.0/addop/Add": "/model.8/m/m.0/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.8/m/m.1/addop/Add": "/model.8/m/m.1/cv2/conv/_input_quantizer/QuantizeLinear",
    "/model.8/m/m.2/addop/Add": "/model.8/m/m.2/cv2/conv/_input_quantizer/QuantizeLinear",
}


LOW_RISK_ADD_TARGETS = [
    "/model.6/m/m.1/addop/Add",
    "/model.6/m/m.4/addop/Add",
    "/model.4/m/m.4/addop/Add",
    "/model.4/m/m.0/addop/Add",
    "/model.6/m/m.0/addop/Add",
]


def build_maps(graph):
    producer = {}
    consumers = defaultdict(list)
    for n in graph.node:
        for out in n.output:
            producer[out] = n
        for idx, inp in enumerate(n.input):
            consumers[inp].append((n, idx))
    return producer, consumers


def find_node_by_name(graph, name):
    for n in graph.node:
        if n.name == name:
            return n
    return None


def clone_attr_name(base, suffix):
    return f"{base}__promote_{suffix}"


def _is_transformer_related(name):
    if not name:
        return False
    return ("/m/tr/" in name) or ("/tr/" in name) or ("/attn/" in name) or ("/mlp/" in name) or ("/norm" in name)


def discover_concat_rules(graph, include_tr):
    producer, consumers = build_maps(graph)
    rules = {}

    for node in graph.node:
        if node.op_type != "Concat":
            continue
        if not node.output:
            continue
        out = node.output[0]
        ql_candidates = []
        for cn, _ in consumers.get(out, []):
            if cn.op_type == "QuantizeLinear" and "_input_quantizer" in (cn.name or ""):
                ql_candidates.append(cn.name)
        if not ql_candidates:
            continue

        if not include_tr:
            tr_related = False
            for inp in node.input:
                prod = producer.get(inp)
                if prod and _is_transformer_related(prod.name):
                    tr_related = True
                    break
            if tr_related:
                continue

        def _score(name):
            if "/cv1/" in name:
                return (0, name)
            if "/cv2/" in name:
                return (1, name)
            if "/cv3/" in name:
                return (2, name)
            return (3, name)

        ql_candidates.sort(key=_score)
        rules[node.name] = ql_candidates[0]

    return rules


def _promote_binary_agg(graph, node_name, ql_ref_name, op_type):
    producer, consumers = build_maps(graph)

    node = find_node_by_name(graph, node_name)
    if node is None or node.op_type != op_type:
        raise RuntimeError(f"{op_type} not found: {node_name}")

    ql_ref = find_node_by_name(graph, ql_ref_name)
    if ql_ref is None or ql_ref.op_type != "QuantizeLinear":
        raise RuntimeError(f"Reference QL not found: {ql_ref_name}")

    if len(ql_ref.input) < 2:
        raise RuntimeError(f"Reference QL missing scale: {ql_ref_name}")
    ref_scale = ql_ref.input[1]
    ref_zero = ql_ref.input[2] if len(ql_ref.input) > 2 else ""

    if not node.output:
        raise RuntimeError(f"{op_type} has no output: {node_name}")
    node_out = node.output[0]
    old_consumers = list(consumers.get(node_out, []))

    new_nodes = []

    for idx, inp in enumerate(node.input):
        q_out = clone_attr_name(node_name.replace("/", "_"), f"in{idx}_q")
        q_name = clone_attr_name(node_name.replace("/", "_"), f"in{idx}_QuantizeLinear")

        q_inputs = [inp, ref_scale]
        if ref_zero:
            q_inputs.append(ref_zero)

        q_node = helper.make_node("QuantizeLinear", q_inputs, [q_out], name=q_name)
        new_nodes.append(q_node)
        node.input[idx] = q_out

    dq_out = clone_attr_name(node_name.replace("/", "_"), "out_dq")
    dq_name = clone_attr_name(node_name.replace("/", "_"), "out_DequantizeLinear")
    dq_inputs = [node_out, ref_scale]
    if ref_zero:
        dq_inputs.append(ref_zero)
    dq_node = helper.make_node("DequantizeLinear", dq_inputs, [dq_out], name=dq_name)
    new_nodes.append(dq_node)

    for node, idx in old_consumers:
        node.input[idx] = dq_out

    nodes_to_remove = set()
    for node, _ in old_consumers:
        if node.op_type != "QuantizeLinear":
            continue
        if "_input_quantizer" not in (node.name or ""):
            continue
        q_out = node.output[0] if node.output else ""
        nodes_to_remove.add(node.name)
        for dq_node_old, _ in consumers.get(q_out, []):
            if dq_node_old.op_type != "DequantizeLinear":
                continue
            dq_out_old = dq_node_old.output[0] if dq_node_old.output else ""
            for dn, didx in consumers.get(dq_out_old, []):
                dn.input[didx] = dq_out
            nodes_to_remove.add(dq_node_old.name)

    if nodes_to_remove:
        kept = [n for n in graph.node if n.name not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(kept)

    for node in reversed(new_nodes):
        graph.node.insert(0, node)

    return len(new_nodes)


def promote_concat(graph, concat_name, ql_ref_name):
    return _promote_binary_agg(graph, concat_name, ql_ref_name, "Concat")


def promote_add(graph, add_name, ql_ref_name):
    return _promote_binary_agg(graph, add_name, ql_ref_name, "Add")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--no-auto", action="store_true")
    parser.add_argument("--include-tr", action="store_true")
    parser.add_argument("--auto-only", nargs="*", default=[])
    parser.add_argument("--add-uniform", action="store_true")
    parser.add_argument("--add-low-risk", action="store_true")
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph

    rules = dict(DEFAULT_RULES)
    if not args.no_auto:
        auto_rules = discover_concat_rules(graph, args.include_tr)
        if args.auto_only:
            auto_rules = {k: v for k, v in auto_rules.items() if k in set(args.auto_only)}
        for k, v in auto_rules.items():
            rules.setdefault(k, v)

    total_new = 0
    for concat_name, ql_ref_name in rules.items():
        total_new += promote_concat(graph, concat_name, ql_ref_name)

    add_targets = []
    if args.add_uniform:
        add_targets.extend(UNIFORM_ADD_SCALE_MAP.keys())
    if args.add_low_risk:
        add_targets.extend(LOW_RISK_ADD_TARGETS)

    for add_name in dict.fromkeys(add_targets):
        ql_ref = UNIFORM_ADD_SCALE_MAP.get(add_name)
        if not ql_ref:
            continue
        total_new += promote_add(graph, add_name, ql_ref)

    onnx.save(model, args.output)
    print(f"Saved: {args.output} ({total_new} nodes inserted)")


if __name__ == "__main__":
    main()
