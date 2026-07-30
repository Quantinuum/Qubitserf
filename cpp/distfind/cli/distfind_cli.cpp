// distfind command-line interface.
//
// Mirrors Quantinuum's Distfind tool: reads Pauli stabiliser strings from stdin
// (one stabiliser per line; chars I/X/Y/Z and . / _ for identity; blank line or EOF
// terminates) and prints the minimum distance of the resulting CSS code. Unlike
// Distfind, distfind only supports CSS codes here, so every stabiliser must be a pure
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
//   -o, --operator PAULI  operator weight of the Pauli string PAULI (may contain Y), given
//                   as a command-line argument; the stabiliser/gauge generators are read
//                   from stdin (one per line). Prints the operator weight
//                   modulo that group (default max(z,x); --zx prints "<z> <x>"). With
//                   --subsystem the generators are the gauge group.
//   --threads N     CPU worker threads (0 => hardware concurrency)
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

#include "distfind/bz.hpp"
#include "distfind/cc.hpp"
#include "distfind/css.hpp"
#include "distfind/gf2.hpp"
#include "distfind/mitm.hpp"
#include "distfind/op_weight.hpp"

using namespace distfind;

namespace {

const char* kUsage =
    "Usage: distfind [OPTIONS]\n"
    "       cat code.txt | distfind [OPTIONS]\n"
    "       distfind --hx Hx.txt --hz Hz.txt [OPTIONS]\n"
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
    "Operator:  -o/--operator PAULI  operator weight of the Pauli string PAULI (may have Y),\n"
    "           given on the command line; generators come from stdin (default max(z,x);\n"
    "           --zx = \"<z> <x>\")\n"
    "Other:    --threads N  --hx FILE  --hz FILE  -v/--verbose  -h/--help\n";

struct Options {
    std::string method = "bz";       // bz | cc | mitm
    std::string backend = "auto";    // auto | cpu | gpu
    char which = 'M';                // M | Z | X
    bool zx = false;                 // print both dZ and dX
    bool subsystem = false;          // treat X/Z input as gauge generators (dressed dist)
    bool operator_mode = false;      // operator-weight mode (-o/--operator PAULI)
    std::string operator_str;        // the operator Pauli string (a command-line argument)
    int threads = 0;
    bool verbose = false;
    std::string hx_path;
    std::string hz_path;
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "distfind: " << msg << "\n";
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
        if (line.empty()) break;  // blank line terminates, like Distfind
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

// Operator-weight input: the stabiliser/gauge generators (CSS: pure X- or Z-type) are read
// from stdin (one per line, blank line / EOF terminates); the operator Pauli (may contain Y)
// is `op_str`, a command-line argument (-o/--operator PAULI).
void read_operator_input(const std::string& op_str, GF2Mat& Gx, GF2Mat& Gz,
                         std::vector<uint8_t>& z_op, std::vector<uint8_t>& x_op, int& n) {
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) break;
        lines.push_back(line);
    }
    if (lines.empty())
        die("operator mode needs at least one generator line on stdin");
    const std::string& op = op_str;

    n = (int)lines[0].size();
    for (const auto& s : lines)
        if ((int)s.size() != n) die("generators have differing lengths");
    if ((int)op.size() != n)
        die("operator length (" + std::to_string(op.size()) +
            ") differs from the generators (" + std::to_string(n) + ")");

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
    opt.verbose = o.verbose;
    return opt;
}

// Distance of one component. Hx/Hz are stabiliser checks for a stabiliser code, or GAUGE
// generators when o.subsystem (then the dressed subsystem distance is computed).
BZResult solve_component(const GF2Mat& Hx, const GF2Mat& Hz, char which, const Options& o) {
    BZOptions opt = make_opt(o);
    auto prob_for = [&](char w) {
        return o.subsystem ? subsystem_problem(Hx, Hz, w) : css_problem(Hx, Hz, w);
    };

    // cc handles Z/X/M itself (with its own weight-level interleave for the min).
    if (o.method == "cc") {
        if (o.subsystem) {
            std::pair<GF2Mat, GF2Mat> center = css_center(Hx, Hz);
            DistProblem pZ = subsystem_problem(Hx, Hz, 'Z');
            DistProblem pX = subsystem_problem(Hx, Hz, 'X');
            return cc_subsystem_distance(center.first, center.second,
                                         pZ.check, pX.check, which, opt);
        }
        return cc_css_distance(Hx, Hz, which, opt);
    }

    if (which == 'Z' || which == 'X') {
        DistProblem p = prob_for(which);
        return (o.method == "mitm") ? mitm_distance(p, opt) : bz_distance(p, opt);
    }
    // min: INTERLEAVE the Z- and X-subproblems weight level by weight level so both lower
    // bounds advance together; a side stalling on a hard level no longer starves the other.
    DistProblem pz = prob_for('Z'), px = prob_for('X');
    return (o.method == "mitm") ? mitm_min_interleaved(pz, px, opt)
                                : bz_min_interleaved(pz, px, opt);
}

// --zx: BOTH distances. Interleaved so both bounds advance together, but UNCAPPED -- each
// side runs to its own full proof, so finding dZ never stops the full dX being found.
std::pair<BZResult, BZResult> solve_zx(const GF2Mat& Hx, const GF2Mat& Hz, const Options& o) {
    BZOptions opt = make_opt(o);
    // cc has no uncapped-pair variant and is fast; compute Z and X separately.
    if (o.method == "cc")
        return {solve_component(Hx, Hz, 'Z', o), solve_component(Hx, Hz, 'X', o)};
    DistProblem pz = o.subsystem ? subsystem_problem(Hx, Hz, 'Z') : css_problem(Hx, Hz, 'Z');
    DistProblem px = o.subsystem ? subsystem_problem(Hx, Hz, 'X') : css_problem(Hx, Hz, 'X');
    return (o.method == "mitm") ? mitm_zx_interleaved(pz, px, opt)
                                : bz_zx_interleaved(pz, px, opt);
}

// Report a non-proven result on stderr so stdout stays just the number.
void note_if_unproven(const BZResult& r, const char* label) {
    if (!r.proven) {
        std::cerr << "distfind: " << label << " d in ["
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
            o.operator_str = need_val(a);   // the operator Pauli is a command-line argument
        } else if (a == "--threads") {
            o.threads = parse_positive_int(a, need_val(a));
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

    // ---- operator weight: generators on stdin + the operator Pauli as a CLI argument ----
    if (o.operator_mode) {
        if (!o.hx_path.empty())
            die("-o/--operator reads generators and the operator from stdin (no --hx/--hz)");
        GF2Mat Gx, Gz;
        std::vector<uint8_t> z_op, x_op;
        int on = 0;
        read_operator_input(o.operator_str, Gx, Gz, z_op, x_op, on);
        if (o.verbose)
            std::cerr << "distfind: operator weight method=" << o.method
                      << " qubits=" << on << " Gx_rows=" << Gx.rows
                      << " Gz_rows=" << Gz.rows << (o.subsystem ? " (gauge)" : "") << "\n";
        BZOptions opt = make_opt(o);
        OpWeight w = css_operator_weight(Gx, Gz, z_op.data(), x_op.data(), on, o.method, opt);
        if (!w.proven)
            std::cerr << "distfind: operator weight not proven (z=" << w.z_weight
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
        std::cerr << "distfind: method=" << o.method << " backend=" << o.backend
                  << " qubits=" << Hx.cols
                  << " Hx_rows=" << Hx.rows << " Hz_rows=" << Hz.rows << "\n";
    }

    if (o.zx) {
        std::pair<BZResult, BZResult> zx = solve_zx(Hx, Hz, o);
        note_if_unproven(zx.first, "dZ:");
        note_if_unproven(zx.second, "dX:");
        std::cout << zx.first.distance << ' ' << zx.second.distance << std::endl;
    } else {
        BZResult r = solve_component(Hx, Hz, o.which, o);
        const char* label = (o.which == 'Z') ? "dZ:" : (o.which == 'X') ? "dX:" : "d:";
        note_if_unproven(r, label);
        std::cout << r.distance << std::endl;
    }
    return 0;
}
