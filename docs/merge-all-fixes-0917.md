# all-fixes 統合報告（2026-09-17）

## 統合対象

- 作業ブランチ: local/all-fixes-0917
- 第1親: e8eeac9（上流 main e6415cc + 大字 c9cf701 + watchdog e8eeac9）
- 第2親: origin/local/all-fixes-upstream / 120f6742538e2fdcaa1da91fae884b570236f5af
- rebase ではなく merge。製品バージョンは 0.11.7 のままです。
- インストール、push、PR、タグ作成は実施していません。

## 衝突23ファイルの解決

「上流」は第1親側、「自前」は第2親側を指します。

| ファイル | 解決内容 |
| --- | --- |
| CHANGELOG.md | 上流を採用。既存リリース記録を保持しました。 |
| apps/rakukan-settings-winui/MainWindow.xaml.cs | 上流のDPI対応と自前のホイール操作修正を統合しました。 |
| apps/rakukan-settings-winui/app.manifest | 上流のDPI宣言を採用しました。 |
| crates/rakukan-dict/src/store.rs | 上流の減衰・検証・v1破棄方針に、自前の直近確定優先、予測、明示削除を統合しました。 |
| crates/rakukan-engine-rpc/src/protocol.rs | 自前のForget/Predict/DictLookupと上流のEngineHealthを統合。EngineHealthを末尾に置き、自前v7の既存variant番号を保持しました。 |
| crates/rakukan-engine/src/conv_cache.rs | 上流の失敗状態管理と自前のユーザー辞書接頭辞変換を統合しました。 |
| crates/rakukan-engine/src/digits.rs | 上流の数値境界検証・大字修正を保持し、自前の助数詞候補を統合しました。 |
| crates/rakukan-engine/src/ffi.rs | 上流のエラー通知に、自前の予測等のAPIとモデル未ロード時の再ロードを統合しました。 |
| crates/rakukan-engine/src/kanji/backend.rs | 上流のエコー検証に、自前の短い読み・オノマトペ保護と不要ASCII除去を統合しました。 |
| crates/rakukan-engine/src/latin_run.rs | 上流を採用。入力ログ設計との整合を優先しました。 |
| crates/rakukan-engine/src/lib.rs | 上流Step10の入力ログとStep12の候補マージに、記号畳み込み、辞書接頭辞、予測抑制、長文リスコア等を移植しました。 |
| crates/rakukan-tsf/src/engine/config.rs | 上流のime_on_apps/ime_off_appsに、自前の予測・リスコア設定を統合しました。 |
| crates/rakukan-tsf/src/engine/keymap.rs | 上流のIME切替・修飾キー処理に、自前のDelete/CandidateForgetを統合しました。 |
| crates/rakukan-tsf/src/engine/state.rs | 上流のModeStoreとpreview_forを保持し、自前の文節状態・伸縮・カーソル途中編集を統合しました。 |
| crates/rakukan-tsf/src/engine/user_action.rs | 上流のモード操作と自前の文節・削除操作を統合しました。 |
| crates/rakukan-tsf/src/tsf/candidate_window.rs | 上流のレイアウト・状態通知・watchdog修正に、自前のstale bg result再試行、予測表示、確定追いつき処理を統合しました。 |
| crates/rakukan-tsf/src/tsf/factory.rs | 上流の構造に、自前の文節表示属性・カーソル・確定再試行を統合。重複したsink解除を除去しました。 |
| crates/rakukan-tsf/src/tsf/factory/dispatch.rs | 上流のモード・preview_forに、自前の文節操作、カーソル移動、予測、確定再試行を統合しました。 |
| crates/rakukan-tsf/src/tsf/factory/edit_ops.rs | 上流のモード遷移と末尾ローマ字順序に、自前の途中編集・文節確定・記号畳み込みを統合しました。 |
| crates/rakukan-tsf/src/tsf/factory/on_compose.rs | 自前120f674の確定再試行を保持し、上流のcaret指定と自前の文節下線表示を共存させました。 |
| crates/rakukan-tsf/src/tsf/factory/on_convert.rs | 自前の文節変換・確定済み先頭保護を、上流の末尾ローマ字・準備状態・host復帰と統合しました。 |
| crates/rakukan-tsf/src/tsf/factory/on_input.rs | 上流の学習判断を保持し、自前の文節全文・末尾文字・途中入力・予測処理を統合しました。 |
| scripts/install.ps1 | 上流の全体手順に、自前のDLL/exe退避リネーム方式を採用しました。実行はしていません。 |

## 採用しなかった自前挙動・置き換え

コミット自体は第2親として履歴に残ります。以下は挙動単位の整理です。

- 903006d: v1学習履歴の頻度付き移行は採用せず、上流のv1履歴破棄方針を採用しました。
- 653dfe7の候補順位部分: LLM候補を辞書より前に固定する独自順位は採用せず、上流Step12の学習→通常ユーザー辞書→辞書→LLM（LLM枠確保）を採用。確定ずれ・文脈エコー対策は残しました。
- 6a1b1ccの確定出力削除後の子音巻き戻し: 上流Step10の打鍵ログ再生に統合。確定済みの素通し子音を後から未確定に戻す挙動は残していません。未確定子音の打鍵再生は維持しました。
- ff2ed4d / 8992f82: text_field_modeによるフォーカス時モード強制は採用せず、上流のアプリ別ime_on_apps/ime_off_appsとModeStoreを採用しました。
- 4061a5b / 5652f57 / e8c1cc3の重複部分: TSF側の独自失敗watchdog・再ロード制御を上流Step13のhost復帰と状態通知に統合しました。0a61fdbのモデル未ロードラッチ解除は保持しています。
- 0c0109fの一律学習強制部分: 上流の候補由来を考慮した学習判断を採用しました。自前の文節・ライブ確定全文の学習処理は残しています。

その他、d633ed1、ddf0d7c、7afe28e、cd3b111、120f674、文節変換・途中編集、および第1親のc9cf701/e8eeac9は保持しました。

## 検証結果

| 検証 | 結果 |
| --- | --- |
| cargo make check | 成功（最終変更後も再確認） |
| cargo make test | 成功。engine 318成功・2 ignored、結合テスト4成功 |
| cargo test -p rakukan-tsf --lib | 146成功 |
| cargo test -p rakukan-dict -p rakukan-engine-rpc --lib | dict 57成功、RPC 20成功 |
| cargo make build-engine | 成功。CPU/Vulkan DLL生成。CUDAはnvcc未導入のため既定スクリプトがスキップ |
| cargo make build-tsf | Rust部分（TSF/tray/host/dict-builder）成功。WinUIの最終処理は環境不足で失敗 |
| conflict marker / git diff --check | 残存・エラーなし |
| 改行/BOM | 既存43変更対象の該当ファイルを検査。既存CRLF/BOMを保持、新規ファイルはLF・BOMなし |

最初のテストで上流と自前の仕様差に対応する2件が失敗しました。学習優先順位とStep10の確定出力Backspaceに合わせて期待値を更新し、全件を再実行して成功しました。

### WinUIビルドの未完了事項

通常のbuild-tsfはMicrosoft.NET.Sdk解決エラーで停止します。SDK 8.0.416は存在しますが、Visual Studio側のSDK resolver/パッケージング構成が不足しています。
一時的なプロセス環境としてMSBuildSDKsPathに既存SDKのSdksを指定し、DOTNET_MSBUILD_SDK_RESOLVER_CLI_DIRをdotnet配置先に、MSBuildEnableWorkloadResolverをfalseにすると、C#・XAMLコンパイルまで進みました。
最終的にはMSB4062: Microsoft.Build.Packaging.Pri.Tasks.ExpandPriContentを読み込めず停止しました。Visual Studio 2022 CommunityのAppxPackage配下にMicrosoft.Build.Packaging.Pri.Tasks.dllが存在しません。リポジトリのビルド設定は変更していません。
環境への追加インストールは行っていないため、build-tsf全体成功の受け入れ条件は未達です。

IMEのインストールと実アプリでの手動入力検証も仕様どおり実施していません。ABIは取り込み元の12、RPCは取り込み元の7で、TSF・host・engineは同じビルド一式で扱う必要があります。旧v7 hostは新規EngineHealth要求を扱えません。

## 作成・変更ファイル一覧

開始HEADとの差分です。Aは新規、Mは変更です。衝突解決で上流をそのまま採用したCHANGELOG/app.manifest/latin_runは差分一覧には含まれません。

- M: apps/rakukan-settings-winui/MainWindow.xaml
- M: apps/rakukan-settings-winui/MainWindow.xaml.cs
- M: crates/rakukan-dict/src/lib.rs
- M: crates/rakukan-dict/src/store.rs
- M: crates/rakukan-dict/src/user_dict.rs
- A: crates/rakukan-dict/tests/learn_probe.rs
- A: crates/rakukan-dict/tests/tail_kana_probe.rs
- M: crates/rakukan-engine-abi/src/lib.rs
- M: crates/rakukan-engine-rpc/src/client.rs
- M: crates/rakukan-engine-rpc/src/protocol.rs
- M: crates/rakukan-engine-rpc/src/server.rs
- M: crates/rakukan-engine/src/conv_cache.rs
- A: crates/rakukan-engine/src/dict_prefix.rs
- M: crates/rakukan-engine/src/digits.rs
- M: crates/rakukan-engine/src/ffi.rs
- M: crates/rakukan-engine/src/kanji/backend.rs
- M: crates/rakukan-engine/src/lib.rs
- A: crates/rakukan-engine/src/rescore.rs
- M: crates/rakukan-engine/src/romaji/converter.rs
- A: crates/rakukan-engine/tests/dict_load_probe.rs
- A: crates/rakukan-engine/tests/reading_probe.rs
- A: crates/rakukan-engine/tests/rescore_probe.rs
- A: crates/rakukan-engine/tests/resegment_probe.rs
- A: crates/rakukan-tsf/src/engine/clause.rs
- M: crates/rakukan-tsf/src/engine/config.rs
- M: crates/rakukan-tsf/src/engine/keymap.rs
- M: crates/rakukan-tsf/src/engine/mod.rs
- M: crates/rakukan-tsf/src/engine/state.rs
- M: crates/rakukan-tsf/src/engine/text_util.rs
- M: crates/rakukan-tsf/src/engine/user_action.rs
- M: crates/rakukan-tsf/src/globals.rs
- M: crates/rakukan-tsf/src/tsf/candidate_window.rs
- M: crates/rakukan-tsf/src/tsf/display_attr.rs
- M: crates/rakukan-tsf/src/tsf/factory.rs
- M: crates/rakukan-tsf/src/tsf/factory/dispatch.rs
- M: crates/rakukan-tsf/src/tsf/factory/edit_ops.rs
- M: crates/rakukan-tsf/src/tsf/factory/on_compose.rs
- M: crates/rakukan-tsf/src/tsf/factory/on_convert.rs
- M: crates/rakukan-tsf/src/tsf/factory/on_input.rs
- M: crates/rakukan-tsf/src/tsf/live_session.rs
- M: crates/rakukan-tsf/src/tsf/mod.rs
- A: crates/rakukan-tsf/src/tsf/suggestion.rs
- M: scripts/install.ps1
- A: docs/merge-all-fixes-0917.md（本報告）
