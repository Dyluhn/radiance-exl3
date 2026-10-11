/* kld_oracle.cpp -- radiance's own KL recorder and loader (core/kld.h), driven from files, so
 * tests/kld/test_kld_synth.py can hold tools/kld_synth.py to the engine's code rather than to a
 * reading of it. Built against a radiance host build (librad_core.a), no device, no model.
 *
 *   kld_oracle record DIR V docs.txt tokens.i32 logits.f32
 *       radiance writes a reference into DIR (Kld::add_doc / record_into / score / finish)
 *   kld_oracle score  DIR V logits.f32 OUT.json
 *       radiance loads the reference in DIR (Kld::score_against -- the --kld-ref loader, with all
 *       its manifest and size checks) and scores the logits against it, writing OUT.json + .rows
 *
 * docs.txt: one "tokens score_from source" line per document. logits.f32: f32 [sum tokens, V],
 * a row at EVERY position of every document in order (Kld::score skips the unscored ones, as the
 * engine hands it whole prefill chunks). Exit 0 on success, the radiance status otherwise.
 */
#include "kld.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace rad;

namespace {

template <class T> std::vector<T> read_all(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { std::fprintf(stderr, "kld_oracle: cannot read %s\n", path.c_str()); std::exit(2); }
    std::vector<T> v((size_t)in.tellg() / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(v.size() * sizeof(T)));
    return v;
}

/* Every document's rows handed over as the engine would: one span per document, its logits
 * block one shard covering the whole vocabulary. */
int score_all(Kld& k, const std::vector<float>& logits, int64_t V) {
    int64_t off = 0;
    for (size_t d = 0; d < k.n_docs(); ++d) {
        const int64_t n = (int64_t)k.tokens(d).size();
        const std::vector<KldSpan> spans = { { d + 1, 0, n, 0 } };
        const std::vector<KldShard> shards = { { logits.data() + off * V, 0, V } };
        if (int st = k.score(spans, shards); st < 0) return st;
        off += n;
    }
    if ((size_t)(off * V) != logits.size()) {
        std::fprintf(stderr, "kld_oracle: logits hold %zu values, the documents need %lld\n",
                     logits.size(), (long long)(off * V));
        return 2;
    }
    return k.finish();
}

}  /* namespace */

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "record" && argc == 7) {
        const int64_t V = std::atoll(argv[3]);
        const std::vector<int32_t> tokens = read_all<int32_t>(argv[5]);
        Kld k;
        std::ifstream docs(argv[4]);
        int64_t n, from, at = 0;
        std::string source;
        while (docs >> n >> from >> source) {
            const std::vector<int32_t> ids(tokens.begin() + at, tokens.begin() + at + n);
            if (int st = k.add_doc(source, ids, from); st < 0) return 3;
            at += n;
        }
        if (int st = k.record_into(argv[2], V, "radiance-recorder"); st < 0) return 4;
        return score_all(k, read_all<float>(argv[6]), V) < 0 ? 5 : 0;
    }
    if (mode == "score" && argc == 6) {
        Kld k;
        if (int st = k.score_against(argv[2], argv[5], std::atoll(argv[3]), "candidate"); st < 0) return 6;
        return score_all(k, read_all<float>(argv[4]), std::atoll(argv[3])) < 0 ? 7 : 0;
    }
    std::fprintf(stderr, "usage: kld_oracle record DIR V docs.txt tokens.i32 logits.f32\n"
                         "       kld_oracle score  DIR V logits.f32 OUT.json\n");
    return 2;
}
