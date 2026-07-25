"""Colored-graph automorphisms via nauty / Traces (system ``dreadnaut``).

The CSS automorphism engines reduce to the automorphism group of a vertex-coloured graph
(a coordinate<->codeword incidence, or a coloured Tanner graph).  Those are solved by nauty's
``dreadnaut`` -- dense **nauty** (``An``) for most graphs, falling back to **Traces** (``At``,
Piperno) for the large, highly-symmetric incidences where nauty's DFS backtracking blows up
(the residual GL(3,2)/A5 group-algebra and big quasi-cyclic families).  ``dreadnaut`` is the
only off-the-shelf tool exposing *both* algorithms, so nauty is a documented system dependency
(``apt install nauty`` / ``conda install -c conda-forge nauty`` / ``brew install nauty``); there
is no pip package that bundles Traces.

This module returns permutation **generators** (0-indexed image lists); group orders/membership
come from :mod:`codeaut.permgroup`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from typing import Optional

import numpy as np

from . import gf2
from . import permgroup


# --------------------------------------------------------------------------- dreadnaut driver

def nauty_binary() -> Optional[str]:
    """Path to nauty's ``dreadnaut`` executable, or ``None`` if not installed."""
    for name in ("dreadnaut", "nauty-dreadnaut"):
        p = shutil.which(name)
        if p:
            return p
    return None


def require_nauty() -> str:
    binp = nauty_binary()
    if binp is None:
        raise RuntimeError(
            "nauty's 'dreadnaut' was not found on PATH.  Install nauty (>= 2.7, includes "
            "Traces): `apt install nauty`, `conda install -c conda-forge nauty`, or "
            "`brew install nauty`.")
    return binp


def dreadnaut_automorphisms(num_vertices: int, adjacency, partition, *, timeout=None,
                            algorithm: str = "nauty"):
    """Generators (full vertex permutations, 0-indexed) of a vertex-coloured graph's
    automorphism group, via ``dreadnaut``.

    ``adjacency``: dict ``v -> list of neighbours``; ``partition``: list of colour cells (lists
    of vertices) the automorphisms must respect.  ``algorithm`` selects the solver on the SAME
    coloured input: ``"nauty"`` (dense, default) or ``"traces"`` (``At`` -- far faster on large,
    highly-symmetric graphs).  ``timeout`` (seconds) caps the subprocess; on expiry a
    :class:`subprocess.TimeoutExpired` is raised.
    """
    binp = require_nauty()
    if algorithm not in ("nauty", "dense", "traces"):
        raise ValueError(f"algorithm must be 'nauty' or 'traces' (got {algorithm!r})")
    lines = []
    if algorithm == "traces":
        lines.append("At")                              # select Traces (sparse) before reading g
    lines.append(f"n={num_vertices} g")
    for v in range(num_vertices):
        nb = " ".join(str(w) for w in sorted(adjacency[v]))
        lines.append(f"{v} : {nb}{';' if v < num_vertices - 1 else '.'}")
    lines.append("f=[" + "|".join(" ".join(str(x) for x in cell) for cell in partition) + "]")
    lines.append("x")
    res = subprocess.run([binp], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, timeout=timeout)
    out = res.stdout
    if res.returncode != 0:
        detail = (res.stderr or out or "no diagnostic output").strip()
        raise RuntimeError(f"dreadnaut exited with status {res.returncode}: {detail}")
    summaries = re.findall(r"grpsize=([^;\s]+);\s*(\d+)\s+gen", out)
    if len(summaries) != 1 or "cpu time" not in out:
        detail = (res.stderr or out or "empty output").strip()
        raise RuntimeError(f"dreadnaut output has no unique completed group summary: {detail}")
    reported_order, reported_generators = summaries[0][0], int(summaries[0][1])
    # nauty prints a generator as a bare "(cycles)" line; Traces as "Gen[(A|M)] #k: (cycles)".
    # Either may wrap mid-cycle onto indented continuation lines (a continuation may start with a
    # digit).  Join indented lines onto the current generator; the cycle regex matches only
    # parenthesised digit runs, so the 'Gen', '#k', '(A)'/'(M)' markers are ignored.
    blocks: list[str] = []
    cur: Optional[str] = None
    for line in out.splitlines():
        if line.startswith("(") or line.startswith("Gen"):
            if cur is not None:
                blocks.append(cur)
            cur = line
        elif cur is not None and line[:1].isspace():
            cur += line
        else:
            if cur is not None:
                blocks.append(cur)
                cur = None
    if cur is not None:
        blocks.append(cur)
    perms = []
    for block in blocks:
        payload = block.split(":", 1)[1] if block.startswith("Gen") and ":" in block else block
        cycles = re.findall(r"\(([\d\s]+)\)", payload)
        leftover = re.sub(r"\([\d\s]+\)", "", payload).strip()
        if leftover:
            raise RuntimeError(f"could not parse complete dreadnaut generator: {block!r}")
        perm = list(range(num_vertices))
        for cyc in cycles:
            pts = [int(t) for t in cyc.split()]
            for i in range(len(pts)):
                perm[pts[i]] = pts[(i + 1) % len(pts)]
        if sorted(perm) != list(range(num_vertices)):
            raise RuntimeError(f"dreadnaut generator is not a permutation: {block!r}")
        perms.append(perm)
    if len(perms) != reported_generators:
        raise RuntimeError(
            f"dreadnaut reported {reported_generators} generators but the parser recovered "
            f"{len(perms)}")

    # This catches subtler parser truncation (for example a wrapped cycle that was only partly
    # consumed).  Dreadnaut prints exact integers for modest orders and a rounded scientific
    # mantissa for large ones; compare the latter to one unit in its last printed place.
    # A full pure-Python Schreier--Sims audit is useful on small parser-regression graphs but can
    # dominate a large solve (S_100 alone takes tens of seconds here).  Above the threshold the
    # strict block-count/completion/syntax/permutation checks establish that every emitted
    # generator was consumed; dreadnaut itself already certified the group summary.
    if num_vertices <= 32:
        actual_order = permgroup.Group(perms, num_vertices).order()
        if "e" in reported_order.lower():
            token = reported_order.lower().split("e", 1)[0]
            precision = sum(char.isdigit() for char in token)
            reported = Decimal(reported_order)
            quantum = Decimal(10) ** (reported.adjusted() - precision + 1)
            with localcontext() as context:
                context.prec = max(precision + 4, 20)
                matches = abs(Decimal(actual_order) - reported) <= quantum
        else:
            matches = actual_order == int(reported_order)
        if not matches:
            raise RuntimeError(
                f"parsed dreadnaut generators have order {actual_order}, but dreadnaut "
                f"reported grpsize={reported_order}")
    return perms


def project_generators(perms, target):
    """Project full-vertex generator permutations onto ``target`` vertices (which a colour cell
    guarantees map among themselves), relabelled to ``0..len(target)-1``.  Returns a list of
    0-indexed image lists on the projected index space."""
    relabel = {v: i for i, v in enumerate(target)}
    t = len(target)
    out = []
    for perm in perms:
        img = [0] * t
        for v in target:
            img[relabel[v]] = relabel[perm[v]]
        out.append(img)
    return out


def _solve_coloured_graph(num_vertices, adjacency, partition, *, timeout=None,
                          nauty_timeout=None, traces_timeout=None):
    """Run the common dense-nauty to Traces ladder on an already built coloured graph."""
    if nauty_timeout is None:
        return dreadnaut_automorphisms(num_vertices, adjacency, partition, timeout=timeout)
    try:
        return dreadnaut_automorphisms(
            num_vertices, adjacency, partition, timeout=nauty_timeout)
    except subprocess.TimeoutExpired:
        return dreadnaut_automorphisms(
            num_vertices, adjacency, partition, algorithm="traces",
            timeout=traces_timeout if traces_timeout is not None else timeout)


def relation_group(vertex_labels, edge_labels, *, timeout=None, nauty_timeout=None,
                   traces_timeout=None):
    """Automorphisms of a labelled complete graph, encoded as a vertex-coloured simple graph.

    ``vertex_labels[i]`` is any hashable coordinate label.  ``edge_labels[i][j]`` is any
    hashable symmetric pair label (the diagonal is ignored).  The most common pair label is
    represented by absence; every other pair becomes a degree-two gadget vertex in a colour
    cell dedicated to its label.  The returned group acts on the original ``n`` coordinates.

    This is the shared encoding for LCD projectors, pair moments, support-splitting signatures,
    and combinations of those invariant relations.
    """
    labels = list(vertex_labels)
    n = len(labels)
    if len(edge_labels) != n or any(len(row) != n for row in edge_labels):
        raise ValueError("edge_labels must be an n x n symmetric table")
    pairs = []
    counts = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            label = edge_labels[i][j]
            if label != edge_labels[j][i]:
                raise ValueError("edge_labels must be symmetric")
            counts[label] += 1
            pairs.append((i, j, label))
    default = counts.most_common(1)[0][0] if counts else None

    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    vertex_cells = defaultdict(list)
    for i, label in enumerate(labels):
        vertex_cells[label].append(i)
    edge_cells = defaultdict(list)
    vid = n
    for i, j, label in pairs:
        if label == default:
            continue
        adjacency[vid] = [i, j]
        adjacency[i].append(vid)
        adjacency[j].append(vid)
        edge_cells[label].append(vid)
        vid += 1
    # Coordinate and relation-gadget cells are deliberately separate even when user labels are
    # equal.  Sorting by repr makes generated dreadnaut input deterministic for heterogeneous
    # tuple/int labels without imposing an ordering contract on labels.
    partition = ([vertex_cells[key] for key in sorted(vertex_cells, key=repr)] +
                 [edge_cells[key] for key in sorted(edge_cells, key=repr)])
    perms = _solve_coloured_graph(
        vid, adjacency, partition, timeout=timeout, nauty_timeout=nauty_timeout,
        traces_timeout=traces_timeout)
    gens = project_generators(perms, list(range(n)))
    return permgroup.Group(gens, n), vid


def hypergraph_group(n: int, vertex_labels, relation_cells, *, timeout=None,
                     nauty_timeout=None, traces_timeout=None):
    """Automorphisms of a coloured hypergraph on ``n`` distinguished coordinate vertices.

    ``relation_cells`` contains ``(label, supports)`` pairs; each support is an iterable of
    coordinate indices and becomes one gadget vertex.  Labels are colour cells, so relations of
    different arity or origin should use different labels.  This generalizes pair-moment graphs
    to guarded triple moments and mixed invariant layers.
    """
    labels = list(vertex_labels)
    if len(labels) != n:
        raise ValueError("vertex_labels must have length n")
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    vertex_cells = defaultdict(list)
    for i, label in enumerate(labels):
        vertex_cells[label].append(i)
    gadget_cells = defaultdict(list)
    vid = n
    for label, supports in relation_cells:
        for support in supports:
            points = sorted(set(int(x) for x in support))
            if any(x < 0 or x >= n for x in points):
                raise ValueError("hyperedge coordinate out of range")
            adjacency[vid] = points
            for point in points:
                adjacency[point].append(vid)
            gadget_cells[label].append(vid)
            vid += 1
    partition = ([vertex_cells[key] for key in sorted(vertex_cells, key=repr)] +
                 [gadget_cells[key] for key in sorted(gadget_cells, key=repr)])
    perms = _solve_coloured_graph(
        vid, adjacency, partition, timeout=timeout, nauty_timeout=nauty_timeout,
        traces_timeout=traces_timeout)
    return permgroup.Group(project_generators(perms, list(range(n))), n), vid


def compressed_incidence_group(n: int, class_groups, *, coordinate_labels=None, timeout=None,
                               traces_timeout=None, nauty_timeout=None):
    """Exact twin quotient of :func:`incidence_group` with the coordinate action lifted back.

    Coordinates having the same initial label and identical incidence neighbourhood are twins.
    They are replaced by one quotient vertex coloured by their multiplicity.  Quotient
    automorphisms lift by matching the ordered coordinates in corresponding twin classes; the
    full symmetric group inside every class is added explicitly.  Thus compression changes graph
    size, not its automorphism group.
    """
    labels = [0] * n if coordinate_labels is None else list(coordinate_labels)
    if len(labels) != n:
        raise ValueError("coordinate_labels must have length n")
    rows_with_colours = []
    colour = 0
    for group in class_groups:
        for _weight, rows in group:
            prepared = [np.asarray(row, dtype=np.uint8).reshape(-1) % 2 for row in rows]
            common_weight = (int(prepared[0].sum()) if prepared and
                             all(int(row.sum()) == int(prepared[0].sum())
                                 for row in prepared) else None)
            use_complement = common_weight is not None and common_weight > n - common_weight
            for row in prepared:
                arr = np.asarray(row, dtype=np.uint8).reshape(-1) % 2
                if arr.size != n:
                    raise ValueError("incidence row has the wrong length")
                if use_complement:
                    arr = 1 - arr
                rows_with_colours.append((arr, colour))
            colour += 1

    signatures = defaultdict(list)
    for point in range(n):
        neighbourhood = tuple(i for i, (row, _cell) in enumerate(rows_with_colours)
                              if row[point])
        signatures[(labels[point], neighbourhood)].append(point)
    twin_classes = sorted(signatures.values(), key=lambda cell: cell[0])
    point_to_type = {}
    for type_index, points in enumerate(twin_classes):
        for point in points:
            point_to_type[point] = type_index

    types = len(twin_classes)
    adjacency: dict[int, list[int]] = {i: [] for i in range(types)}
    type_cells = defaultdict(list)
    for type_index, points in enumerate(twin_classes):
        type_cells[(labels[points[0]], len(points))].append(type_index)
    word_cells = defaultdict(list)
    quotient_edges = 0
    vid = types
    for row_index, (row, colour_index) in enumerate(rows_with_colours):
        incident_types = sorted({point_to_type[int(p)] for p in np.flatnonzero(row)})
        quotient_edges += len(incident_types)
        adjacency[vid] = incident_types
        for type_index in incident_types:
            adjacency[type_index].append(vid)
        word_cells[colour_index].append(vid)
        vid += 1
    partition = ([type_cells[key] for key in sorted(type_cells, key=repr)] +
                 [word_cells[key] for key in sorted(word_cells)])
    perms = _solve_coloured_graph(
        vid, adjacency, partition, timeout=timeout, nauty_timeout=nauty_timeout,
        traces_timeout=traces_timeout)

    generators = []
    for full_perm in perms:
        lifted = list(range(n))
        for source_type, source_points in enumerate(twin_classes):
            target_type = full_perm[source_type]
            target_points = twin_classes[target_type]
            if len(source_points) != len(target_points):
                raise AssertionError("twin quotient mapped unequal multiplicities")
            for source, target in zip(source_points, target_points):
                lifted[source] = target
        generators.append(lifted)
    for points in twin_classes:
        for index in range(len(points) - 1):
            transposition = list(range(n))
            a, b = points[index:index + 2]
            transposition[a], transposition[b] = b, a
            generators.append(transposition)
    return permgroup.Group(generators, n), vid, len(twin_classes), quotient_edges


def incidence_group(n: int, class_groups, *, timeout=None, traces_timeout=None,
                    nauty_timeout=None):
    """Automorphism group (a :class:`codeaut.permgroup.Group` on the ``n`` coordinates) of the
    coloured coordinate<->codeword incidence built from one or more groups of weight classes.

    ``class_groups`` is an iterable of iterables of ``(weight, rows)`` (e.g. one per CSS side);
    coordinate vertices ``0..n-1`` form one colour, and every weight class becomes its own
    disjoint colour cell.  Solver ladder: dense nauty; if ``nauty_timeout`` is set and nauty
    does not finish in time, fall back to Traces (capped at ``traces_timeout``).  Returns
    ``(group, num_vertices)``.
    """
    adjacency: dict[int, list[int]] = {v: [] for v in range(n)}
    col_cells: list[list[int]] = [list(range(n))]
    vid = n
    for grp in class_groups:
        for _w, rows in grp:
            cell = []
            for row in rows:
                nb = [int(p) for p in np.flatnonzero(row)]
                adjacency[vid] = list(nb)
                for p in nb:
                    adjacency[p].append(vid)
                cell.append(vid)
                vid += 1
            if cell:
                col_cells.append(cell)
    if nauty_timeout is None:
        perms = dreadnaut_automorphisms(vid, adjacency, col_cells, timeout=timeout)
    else:
        try:
            perms = dreadnaut_automorphisms(vid, adjacency, col_cells, timeout=nauty_timeout)
        except subprocess.TimeoutExpired:
            perms = dreadnaut_automorphisms(vid, adjacency, col_cells, algorithm="traces",
                                            timeout=traces_timeout)
    gens = project_generators(perms, list(range(n)))
    return permgroup.Group(gens, n), vid


# ----------------------------------------------------------- parity-check / Tanner automorphisms

def parity_check_automorphism_group(Hx, Hz, *, algorithm: str = "nauty", timeout=None):
    """Type-preserving **qubit-permutation** automorphism group of a CSS code's parity-check
    (Tanner) graph, via graph automorphism (nauty/Traces) -- a sound subgroup of the full code
    automorphism group.  These are the qubit permutations preserving both ``Hx`` and ``Hz`` as
    row multisets; the result is a :class:`codeaut.permgroup.Group` on the ``n`` qubits.

    The graph has the ``2n`` columns of ``[Z | X]`` (X-stabilisers ``[0 | Hx]``, Z-stabilisers
    ``[Hz | 0]``), one vertex per stabiliser row, and one gadget vertex per qubit tying its Z- and
    X-columns together so a single qubit permutation acts on both blocks.  Z- and X-columns are
    kept in separate colour cells (type-preserving), and the automorphisms are projected onto the
    ``n`` qubits.  ``algorithm`` is ``"nauty"`` (dense, default) or ``"traces"`` (sparse; faster
    on large, highly symmetric graphs).
    """
    Hx = gf2.as_uint8(Hx)
    Hz = gf2.as_uint8(Hz)
    n = Hx.shape[1]
    # [z | x] stabiliser rows: X-stabilisers -> [0 | Hx], Z-stabilisers -> [Hz | 0]
    zero_x = np.zeros((Hx.shape[0], n), dtype=np.uint8)
    zero_z = np.zeros((Hz.shape[0], n), dtype=np.uint8)
    stabs = np.vstack([np.hstack([zero_x, Hx]), np.hstack([Hz, zero_z])]).astype(np.uint8)
    m = stabs.shape[0]
    ncols = 2 * n
    pairs = [(q, n + q) for q in range(n)]                    # tie each qubit's Z- and X-column
    col_cells = [list(range(n)), list(range(n, 2 * n))]       # Z- and X-slots kept apart
    target = list(range(n))                                   # project to qubits

    crow0, brow0 = ncols, ncols + m
    V = ncols + m + n
    adjacency = {v: [] for v in range(V)}
    for i in range(m):
        for j in np.flatnonzero(stabs[i]):
            adjacency[crow0 + i].append(int(j))
            adjacency[int(j)].append(crow0 + i)
    for q, pair in enumerate(pairs):
        for j in pair:
            adjacency[brow0 + q].append(j)
            adjacency[j].append(brow0 + q)
    partition = col_cells + [list(range(crow0, crow0 + m)), list(range(brow0, brow0 + n))]

    perms = dreadnaut_automorphisms(V, adjacency, partition, algorithm=algorithm, timeout=timeout)
    gens = project_generators(perms, target)
    return permgroup.Group(gens, len(target))


def tanner_permutation_group(Hx, Hz, *, algorithm: str = "nauty", timeout=None):
    """Type-preserving qubit-permutation group (preserving ``Hx`` and ``Hz`` row multisets) as a
    :class:`codeaut.permgroup.Group` on ``n`` points -- alias of
    :func:`parity_check_automorphism_group`."""
    return parity_check_automorphism_group(Hx, Hz, algorithm=algorithm, timeout=timeout)


def matrix_tanner_group(H, *, algorithm: str = "nauty", timeout=None):
    """Automorphism group of the **bipartite Tanner graph** of a single GF(2) parity-check (or
    generator) matrix ``H`` (``m x n``): variable nodes ``0..n-1`` on one side, check nodes on the
    other, an edge per nonzero entry.  Returns a :class:`codeaut.permgroup.Group` on the ``n``
    variable nodes (column permutations preserving the check structure).

    This is a *graph* symmetry of the chosen ``H`` -- a sound subgroup of the true code
    automorphism group, and basis-dependent (unlike Leon's codeword-based
    :func:`codeaut.code_automorphism_group`)."""
    H = gf2.as_uint8(H)
    m, n = H.shape
    V = n + m
    adjacency = {v: [] for v in range(V)}
    for i in range(m):
        for j in np.flatnonzero(H[i]):
            adjacency[n + i].append(int(j))
            adjacency[int(j)].append(n + i)
    partition = [list(range(n)), list(range(n, n + m))]   # variables and checks kept apart
    perms = dreadnaut_automorphisms(V, adjacency, partition, algorithm=algorithm, timeout=timeout)
    gens = project_generators(perms, list(range(n)))
    return permgroup.Group(gens, n)
