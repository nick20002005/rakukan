#[test]
fn probe_resegment_boundaries() {
    let store = match rakukan_engine::dict::loader::load_dict() {
        rakukan_engine::dict::loader::LoadResult::Ok(s) => s,
        rakukan_engine::dict::loader::LoadResult::Failed { step, reason } => {
            println!("FAILED step={step} reason={reason}");
            return;
        }
    };
    for full in [
        "しじぶんもそんなにこまかくはなさそうだし",
        "わたしはがっこうへいく",
        "これならもっとてんじょうよりにならないとだよ",
    ] {
        let chars: Vec<char> = full.chars().collect();
        let n = chars.len();
        let mut bounds: Vec<usize> = Vec::new();
        for len in 1..n {
            let key: String = chars[..len].iter().collect();
            let hit = !store.lookup_user(&key).is_empty()
                || !store.lookup_dict(&key, 1).is_empty();
            if hit {
                bounds.push(len);
            }
        }
        println!("{full} (n={n}) boundaries={bounds:?}");
        for len in bounds.iter().rev().take(3) {
            let key: String = chars[..*len].iter().collect();
            println!("   {len}: {key} -> {:?}", store.lookup_dict(&key, 3));
        }
    }
}
