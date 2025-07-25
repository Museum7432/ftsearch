#include "SearchResult.h"

namespace seqr {
ItemScore::ItemScore(float score_, size_t id_)
    : Score{score_}, id_(id_) {}

SeqScore::SeqScore(float score_, std::vector<size_t> ids_)
    : Score{score_}, ids_(ids_) {}

} // namespace seqr