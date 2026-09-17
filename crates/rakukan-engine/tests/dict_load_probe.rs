#[test]
fn probe_dict_load() {
    match rakukan_engine::dict::loader::load_dict() {
        rakukan_engine::dict::loader::LoadResult::Ok(store) => {
            println!("user_entries = {}", store.user_entry_count());
            for r in [
                "さんかく",
                "くろさんかく",
                "かぎかっこ",
                "やじるし",
                "まるいち",
            ] {
                println!("  lookup_user({r}) = {:?}", store.lookup_user(r));
                println!("  lookup      ({r}) = {:?}", store.lookup(r, 8).candidates);
            }
        }
        rakukan_engine::dict::loader::LoadResult::Failed { step, reason } => {
            println!("FAILED step={step} reason={reason}");
        }
    }
}
