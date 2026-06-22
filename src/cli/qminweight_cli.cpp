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
    "Other:    --threads N  --max-weight N  --hx FILE  --hz FILE  -v/--verbose  -h/--help\n";

struct Options {
    std::string method = "bz";       // bz | cc | mitm
    std::string backend = "auto";    // auto | cpu | gpu
    char which = 'M';                // M | Z | X
    bool zx = false;                 // print both dZ and dX
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

BZResult solve_component(const GF2Mat& Hx, const GF2Mat& Hz, char which, const Options& o) {
    BZOptions opt;
    opt.backend = o.backend;
    opt.threads = o.threads;
    opt.max_weight = o.max_weight;
    opt.verbose = o.verbose;

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
