//! 実辞書（MOZC + ユーザー辞書 + 学習履歴）で、長文候補の辞書一致スコアが
//! どれくらい開くかを観測する。`[conversion] rescore_min_gain` を決め直す
//! ときの材料。辞書が読めない環境では黙って skip する（他の probe と同じ）。
//!
//! `cargo test -p rakukan-engine --test rescore_probe -- --nocapture`

use rakukan_engine::rescore::dict_agreement_score;

#[test]
fn probe_dict_agreement_scores() {
    let store = match rakukan_engine::dict::loader::load_dict() {
        rakukan_engine::dict::loader::LoadResult::Ok(s) => s,
        rakukan_engine::dict::loader::LoadResult::Failed { step, reason } => {
            println!("FAILED step={step} reason={reason}");
            return;
        }
    };

    // (読み, [候補...])。先頭が「拾いたい正解」とは限らない。差の開き方を見る。
    let cases: [(&str, &[&str]); 4] = [
        (
            "しじぶんもそんなにこまかくはなさそうだし",
            &[
                "指示文もそんなに細かくはなさそうだし",
                "指示分もそんなに細かくはなさそうだし",
                "しじぶんもそんなに細かくはなさそうだし",
            ],
        ),
        (
            "らいしゅうのかいぎしりょうをかくにんする",
            &[
                "来週の会議資料を確認する",
                "来週の会議しりょうを確認する",
                "来週の回議資料を確認する",
            ],
        ),
        (
            "れーるがんいますぐにつくれない",
            &["レールガン今すぐに作れない", "れーるがん今すぐに作れない"],
        ),
        (
            // 文脈でしか決まらない曖昧さ。ここが min_gain 未満に収まっていること。
            "きょうはいしゃにいくよていです",
            &["今日は医者に行く予定です", "今日歯医者に行く予定です"],
        ),
    ];

    for (reading, candidates) in cases {
        println!("--- {reading}");
        for c in candidates {
            println!("    {:?}\t{c}", dict_agreement_score(&store, reading, c));
        }
    }
}
