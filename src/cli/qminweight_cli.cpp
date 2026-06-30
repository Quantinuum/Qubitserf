// qminweight command-line interface.
//
// Mirrors Quantinuum's Qubitserf tool: reads Pauli stabiliser strings from stdin
// (one stabiliser per line; chars I/X/Y/Z and . / _ for identity; blank line or EOF
// terminates) and prints the minimum distance of the resulting CSS code. Unlike
// Qubitserf, qminweight only supports CSS codes here, so every stabiliser must be a pure
// X-type or pure Z-type operator (no Y, no mixed X/Z on one line).
//
// Flags:
//   --bz            Brouwer-Zimmermann method (default)
//   --cc            connected-cluster method (best for sparse / LDPC codes)
//   --mitm          meet-in-the-middle method
//   --cpu/--gpu      backend selection (default: auto)
//   --zx            print "<dZ> <dX>" instead of the single minimum distance
//   --z / --x       compute only the Z- or X-distance (single integer)
//   --subsystem     treat the input X/Z operators as GAUGE generators of a subsystem code
//                   and report the dressed subsystem distance (stabilizer center computed
//                   internally). Combines with --z/--x/--zx/--method/...
//   -o, --operator  the LAST stdin Pauli line is an OPERATOR (may contain Y); the preceding
//                   lines are the stabiliser/gauge generators. Prints the operator weight
//                   modulo that group (default max(z,x); --zx prints "<z> <x>"). With
//                   --subsystem the generators are the gauge group.
//   --threads N     CPU worker threads (0 => hardware concurrency)
//   --max-weight N  safety cap on the enumeration weight (0 => none)
//   --hx FILE --hz FILE   read Hx and Hz from 0/1 text matrices instead of stdin
//   -v, --verbose   verbose diagnostics on stderr
//   -h, --help      this help
//
// stdout stays parseable: it is just the number(s). Non-proven results and warnings go
// to stderr.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "qminweight/bz.hpp"
#include "qminweight/cc.hpp"
#include "qminweight/css.hpp"
#include "qminweight/gf2.hpp"
#include "qminweight/mitm.hpp"
#include "qminweight/op_weight.hpp"

using namespace qminweight;

namespace {

const char* kUsage =
    "Usage: qminweight [OPTIONS]\n"
    "       cat code.txt | qminweight [OPTIONS]\n"
    "       qminweight --hx Hx.txt --hz Hz.txt [OPTIONS]\n"
    "\n"
    "Compute the minimum distance of a CSS code.\n"
    "\n"
    "Input (default): Pauli stabiliser strings on stdin, one per line. Characters\n"
    "  I . _   identity\n"
    "  X       Pauli X      Z   Pauli Z\n"
    "A blank line or EOF terminates input. Each stabiliser must be pure X-type or pure\n"
    "Z-type (Y or mixed X/Z is rejected -- only CSS codes are supported).\n"
    "\n"
    "Method:   --bz (default)  --cc  --mitm\n"
    "Backend:  --cpu  --gpu                 (default: auto; --gpu auto-detects accelerator)\n"
    "Output:   default = minimum distance; --zx = \"<dZ> <dX>\"; --z / --x = one value\n"
    "Subsystem: --subsystem  treat X/Z input as GAUGE generators; report dressed distance\n"
    "Operator:  -o/--operator  last stdin Pauli line is an operator (may have Y); print its\n"
    "           weight modulo the group (default max(z,x); --zx = \"<z> <x>\")\n"
    "Other:    --threads N  --max-weight N  --hx FILE  --hz FILE  -v/--verbose  -h/--help\n";

struct Options {
    std::string method = "bz";       // bz | cc | mitm
    std::string backend = "auto";    // auto | cpu | gpu
    char which = 'M';                // M | Z | X
    bool zx = false;                 // print both dZ and dX
    bool subsystem = false;          // treat X/Z input as gauge generators (dressed dist)
    bool operator_mode = false;      // last stdin line is an operator (operator weight)
    int threads = 0;
    int max_weight = 0;
    bool verbose = false;
    std::string hx_path;
    std::string hz_path;
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "qminweight: " << msg << "\n";
    std::exit(2);
}

int parse_positive_int(const std::string& flag, const std::string& val) {
    if (val.empty() || !std::all_of(val.begin(), val.end(), [](unsigned char c){ return std::isdigit(c); }))
        die(flag + " expects a non-negative integer, got '" + val + "'");
    return std::stoi(val);
}

// Parse a whitespace-separated 0/1 text matrix (one matrix row per line, blank lines
// ignored). Used for --hx / --hz files.
GF2Mat read_matrix_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open file: " + path);
    std::vector<std::vector<uint8_t>> rows;
    std::string line;
    int cols = -1;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::vector<uint8_t> row;
        std::string tok;
        while (ls >> tok) {
            if (tok == "0") row.push_back(0);
            else if (tok == "1") row.push_back(1);
            else die("matrix file " + path + " contains non-binary token '" + tok + "'");
        }
        if (row.empty()) continue;  // skip blank lines
        if (cols < 0) cols = (int)row.size();
        else if ((int)row.size() != cols)
            die("ragged matrix in " + path + " (rows have differing widths)");
        rows.push_back(std::move(row));
    }
    if (rows.empty()) die("matrix file " + path + " is empty");
    std::vector<uint8_t> flat;
    flat.reserve(rows.size() * (size_t)cols);
    for (auto& r : rows)
        for (uint8_t b : r) flat.push_back(b);
    return from_dense(flat.data(), (int)rows.size(), cols);
}

// Read Pauli stabiliser strings from stdin and split into Hx (X-type rows) and Hz
// (Z-type rows). Returns the number of physical qubits in *n. Rejects non-CSS input.
void read_pauli_stdin(GF2Mat& Hx, GF2Mat& Hz, int& n) {
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(std::cin, line)) {
        // strip a trailing carriage return (Windows line endings)
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) break;  // blank line terminates, like Qubitserf
        lines.push_back(line);
    }
    if (lines.empty()) die("no stabilisers on stdin (provide Pauli strings or --hx/--hz)");

    n = (int)lines[0].size();
    for (const auto& s : lines)
        if ((int)s.size() != n)
            die("stabilisers have differing lengths (all rows must span the same qubits)");

    std::vector<std::vector<uint8_t>> xrows, zrows;
    for (size_t li = 0; li < lines.size(); ++li) {
        const std::string& s = lines[li];
        std::vector<uint8_t> xrow(n, 0), zrow(n, 0);
        bool has_x = false, has_z = false;
        for (int j = 0; j < n; ++j) {
            switch (s[j]) {
                case 'I': case '.': case '_': case ' ':
                    break;
                case 'X': case 'x':
                    xrow[j] = 1; has_x = true; break;
                case 'Z': case 'z':
                    zrow[j] = 1; has_z = true; break;
                case 'Y': case 'y':
                    die("stabiliser " + std::to_string(li + 1) +
                        " contains a Y -- only CSS codes are supported");
                default:
                    die(std::string("stabiliser ") + std::to_string(li + 1) +
                        " has an unrecognised character '" + s[j] + "'");
            }
        }
        if (has_x && has_z)
            die("stabiliser " + std::to_string(li + 1) +
                " mixes X and Z -- only CSS codes are supported");
        if (has_x) xrows.push_back(std::move(xrow));
        else if (has_z) zrows.push_back(std::move(zrow));
        // an all-identity row contributes nothing; silently ignore it
    }

    auto build = [n](const std::vector<std::vector<uint8_t>>& rows) -> GF2Mat {
        if (rows.empty()) return GF2Mat(0, n);
        std::vector<uint8_t> flat;
        flat.reserve(rows.size() * (size_t)n);
        for (auto& r : rows)
            for (uint8_t b : r) flat.push_back(b);
        return from_dense(flat.data(), (int)rows.size(), n);
    };
    Hx = build(xrows);
    Hz = build(zrows);
}

// Parse one Pauli operator line of length n into Z-support and X-support (Y sets both).
void parse_operator_line(const std::string& s, int n,
                         std::vector<uint8_t>& z_op, std::vector<uint8_t>& x_op) {
    z_op.assign(n, 0);
    x_op.assign(n, 0);
    for (int j = 0; j < n; ++j) {
        switch (s[j]) {
            case 'I': case '.': case '_': case ' ': break;
            case 'X': case 'x': x_op[j] = 1; break;
            case 'Z': case 'z': z_op[j] = 1; break;
            case 'Y': case 'y': x_op[j] = 1; z_op[j] = 1; break;
            default:
                die(std::string("operator has an unrecognised character '") + s[j] + "'");
        }
    }
}

// Operator-weight input: the LAST stdin Pauli line is the operator (may contain Y); the
// preceding lines are the stabiliser/gauge generators (CSS: pure X- or Z-type).
void read_operator_stdin(GF2Mat& Gx, GF2Mat& Gz,
                         std::vector<uint8_t>& z_op, std::vector<uint8_t>& x_op, int& n) {
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) break;
        lines.push_back(line);
    }
    if (lines.size() < 2)
        die("operator mode needs at least one generator line and one operator line");
    std::string op = lines.back();
    lines.pop_back();

    n = (int)lines[0].size();
    for (const auto& s : lines)
        if ((int)s.size() != n) die("generators have differing lengths");
    if ((int)op.size() != n) die("operator length differs from the generators");

    std::vector<std::vector<uint8_t>> xrows, zrows;
    for (size_t li = 0; li < lines.size(); ++li) {
        const std::string& s = lines[li];
        std::vector<uint8_t> xrow(n, 0), zrow(n, 0);
        bool has_x = false, has_z = false;
        for (int j = 0; j < n; ++j) {
            switch (s[j]) {
                case 'I': case '.': case '_': case ' ': break;
                case 'X': case 'x': xrow[j] = 1; has_x = true; break;
                case 'Z': case 'z': zrow[j] = 1; has_z = true; break;
                case 'Y': case 'y':
                    die("generator " + std::to_string(li + 1) +
                        " contains a Y -- generators must be pure X- or Z-type (CSS)");
                default:
                    die(std::string("generator ") + std::to_string(li + 1) +
                        " has an unrecognised character '" + s[j] + "'");
            }
        }
        if (has_x && has_z)
            die("generator " + std::to_string(li + 1) + " mixes X and Z (CSS only)");
        if (has_x) xrows.push_back(std::move(xrow));
        else if (has_z) zrows.push_back(std::move(zrow));
    }
    auto build = [n](const std::vector<std::vector<uint8_t>>& rows) -> GF2Mat {
        if (rows.empty()) return GF2Mat(0, n);
        std::vector<uint8_t> flat;
        flat.reserve(rows.size() * (size_t)n);
        for (auto& r : rows)
            for (uint8_t b : r) flat.push_back(b);
        return from_dense(flat.data(), (int)rows.size(), n);
    };
    Gx = build(xrows);
    Gz = build(zrows);
    parse_operator_line(op, n, z_op, x_op);
}

BZOptions make_opt(const Options& o) {
    BZOptions opt;
    opt.backend = o.backend;
    opt.threads = o.threads;
    opt.max_weight = o.max_weight;
    opt.verbose = o.verbose;
    return opt;
}

// Distance of one component. Hx/Hz are stabiliser checks for a stabiliser code, or GAUGE
// generators when o.subsystem (then the dressed subsystem distance is computed).
BZResult solve_component(const GF2Mat& Hx, const GF2Mat& Hz, char which, const Options& o) {
    BZOptions opt = make_opt(o);

    if (o.subsystem) {
        std::pair<GF2Mat, GF2Mat> center = css_center(Hx, Hz);
        if (o.method == "cc") {
            DistProblem pZ = subsystem_problem(Hx, Hz, 'Z');
            DistProblem pX = subsystem_problem(Hx, Hz, 'X');
            return cc_subsystem_distance(center.first, center.second,
                                         pZ.check, pX.check, which, opt);
        }
        auto run = [&](char w) -> BZResult {
            DistProblem prob = subsystem_problem(Hx, Hz, w);
            return (o.method == "mitm") ? mitm_distance(prob, opt) : bz_distance(prob, opt);
        };
        if (which == 'Z') return run('Z');
        if (which == 'X') return run('X');
        BZResult z = run('Z'), x = run('X');   // 'M' = true min over Z and X
        auto v = [](int d) { return d < 0 ? (1 << 30) : d; };
        BZResult r = (v(x.distance) < v(z.distance)) ? x : z;
        r.distance = std::min(v(z.distance), v(x.distance));
        if (r.distance >= (1 << 30)) r.distance = -1;
        r.lower_bound = std::min(z.lower_bound, x.lower_bound);
        r.seconds = z.seconds + x.seconds;
        r.proven = z.proven && x.proven;
        return r;
    }

    if (o.method == "cc")
        return cc_css_distance(Hx, Hz, which, opt);

    DistProblem prob = css_problem(Hx, Hz, which);
    if (o.method == "mitm")
        return mitm_distance(prob, opt);
    return bz_distance(prob, opt);
}

// Report a non-proven result on stderr so stdout stays just the number.
void note_if_unproven(const BZResult& r, const char* label) {
    if (!r.proven) {
        std::cerr << "qminweight: " << label << " d in ["
                  << r.lower_bound << ", " << r.distance
                  << "] (not proven)\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options o;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_val = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) die(flag + " requires a value");
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            std::cout << kUsage;
            return 0;
        } else if (a == "--bz") {
            o.method = "bz";
        } else if (a == "--cc") {
            o.method = "cc";
        } else if (a == "--mitm") {
            o.method = "mitm";
        } else if (a == "--cpu") {
            o.backend = "cpu";
        } else if (a == "--gpu") {
            o.backend = "gpu";
        } else if (a == "--zx") {
            o.zx = true;
        } else if (a == "--z") {
            o.which = 'Z';
        } else if (a == "--x") {
            o.which = 'X';
        } else if (a == "--subsystem") {
            o.subsystem = true;
        } else if (a == "-o" || a == "--operator") {
            o.operator_mode = true;
        } else if (a == "--threads") {
            o.threads = parse_positive_int(a, need_val(a));
        } else if (a == "--max-weight") {
            o.max_weight = parse_positive_int(a, need_val(a));
        } else if (a == "-v" || a == "--verbose") {
            o.verbose = true;
        } else if (a == "--hx") {
            o.hx_path = need_val(a);
        } else if (a == "--hz") {
            o.hz_path = need_val(a);
        } else {
            die("unrecognised argument '" + a + "' (try --help)");
        }
    }

    if (o.zx && (o.which != 'M'))
        die("--zx cannot be combined with --z or --x");
    if ((o.hx_path.empty()) != (o.hz_path.empty()))
        die("--hx and --hz must be given together");

    // ---- operator weight: generators + a trailing operator line on stdin ----
    if (o.operator_mode) {
        if (!o.hx_path.empty())
            die("-o/--operator reads generators and the operator from stdin (no --hx/--hz)");
        GF2Mat Gx, Gz;
        std::vector<uint8_t> z_op, x_op;
        int on = 0;
        read_operator_stdin(Gx, Gz, z_op, x_op, on);
        if (o.verbose)
            std::cerr << "qminweight: operator weight method=" << o.method
                      << " qubits=" << on << " Gx_rows=" << Gx.rows
                      << " Gz_rows=" << Gz.rows << (o.subsystem ? " (gauge)" : "") << "\n";
        BZOptions opt = make_opt(o);
        OpWeight w = css_operator_weight(Gx, Gz, z_op.data(), x_op.data(), on, o.method, opt);
        if (!w.proven)
            std::cerr << "qminweight: operator weight not proven (z=" << w.z_weight
                      << ", x=" << w.x_weight << ")\n";
        if (o.zx) std::cout << w.z_weight << ' ' << w.x_weight << std::endl;
        else      std::cout << std::max(w.z_weight, w.x_weight) << std::endl;
        return 0;
    }

    GF2Mat Hx, Hz;
    int n = 0;
    if (!o.hx_path.empty()) {
        Hx = read_matrix_file(o.hx_path);
        Hz = read_matrix_file(o.hz_path);
        if (Hx.cols != Hz.cols)
            die("Hx and Hz have differing column counts (different number of qubits)");
        n = Hx.cols;
    } else {
        read_pauli_stdin(Hx, Hz, n);
    }
    (void)n;

    if (o.verbose) {
        std::cerr << "qminweight: method=" << o.method << " backend=" << o.backend
                  << " qubits=" << Hx.cols
                  << " Hx_rows=" << Hx.rows << " Hz_rows=" << Hz.rows << "\n";
    }

    if (o.zx) {
        BZResult z = solve_component(Hx, Hz, 'Z', o);
        BZResult x = solve_component(Hx, Hz, 'X', o);
        note_if_unproven(z, "dZ:");
        note_if_unproven(x, "dX:");
        std::cout << z.distance << ' ' << x.distance << std::endl;
    } else {
        BZResult r = solve_component(Hx, Hz, o.which, o);
        const char* label = (o.which == 'Z') ? "dZ:" : (o.which == 'X') ? "dX:" : "d:";
        note_if_unproven(r, label);
        std::cout << r.distance << std::endl;
    }
    return 0;
}
