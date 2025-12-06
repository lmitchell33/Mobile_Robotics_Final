import os

# GTSAM factor greaph viewer
# converts graph to a text based graph used by graphviz, uses graphviz to render PDF
def view_factor_graph(graph, values, savePath="factor_graph.pdf"):
    try:
        import gtsam
        from gtsam.utils import graphviz
    except ImportError:
        print("\n[GTSAM Viewer] GTSAM not installed. Skipping factor-graph rendering.\n")
        return

    # Convert graph to DOT string
    dotStr = graphviz.to_dot(graph, values)

    dotPath = savePath.replace(".pdf", ".dot")
    with open(dotPath, "w") as f:
        f.write(dotStr)

    # Attempt to render with graphviz
    cmd = f"dot -Tpdf {dotPath} -o {savePath}"
    print(f"[GTSAM Viewer] Running: {cmd}")
    os.system(cmd)

    print(f"[GTSAM Viewer] Saved factor graph to {savePath}")