# Gate Trust Model

The identity gate authorizes specific runtime optimizations after testing
them. A `PASS` applies only to the tested combination, bounded artifact
fingerprints, ComfyUI commit, dgx-monarch gate protocol, package version,
canonical package-source manifest, and capability context. It does not
establish that every use of the checkpoint or family is safe. The
identity fields and lookup selectors are enforced in
[`gate_ledger.py:105-123`](../src/dgx_monarch/gate_ledger.py#L105-L123 "anchor:artifact_set_signature"),
[`gate_ledger.py:149-152`](../src/dgx_monarch/gate_ledger.py#L149-L152 "anchor:combo_key"),
[`gate_ledger.py:62-69`](../src/dgx_monarch/gate_ledger.py#L62-L69 "anchor:gate_verdict_token"), and
[`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity"), with
coverage in [`test_gate_ledger.py:23-27`](../tests/test_gate_ledger.py#L23-L27 "anchor:test_combo_key_ignores_lora_order_and_detects_name_change"),
[`test_gate_ledger.py:132-136`](../tests/test_gate_ledger.py#L132-L136 "anchor:test_artifact_set_signature_binds_every_entry_without_truncation"),
[`test_gate_ledger.py:139-144`](../tests/test_gate_ledger.py#L139-L144 "anchor:test_artifact_set_signature_is_an_explicit_copyable_value"),
[`test_gate_ledger.py:180-188`](../tests/test_gate_ledger.py#L180-L188 "anchor:test_lookup_accepts_complete_pre_upgrade_artifact_aggregate"),
[`test_gate_ledger.py:191-205`](../tests/test_gate_ledger.py#L191-L205 "anchor:test_incomplete_legacy_aggregate_preserves_fail_but_not_pass"),
[`test_gate_ledger.py:215-228`](../tests/test_gate_ledger.py#L215-L228 "anchor:test_lookup_semantics"),
[`test_gate_ledger.py:247-269`](../tests/test_gate_ledger.py#L247-L269 "anchor:test_pass_is_bound_to_protocol_version_and_capability_context"),
[`test_gate_ledger.py:311-343`](../tests/test_gate_ledger.py#L311-L343 "anchor:test_burned_v4_rows_read_stale_not_current_trust"),
[`test_gate_ledger.py:346-350`](../tests/test_gate_ledger.py#L346-L350 "anchor:test_recorded_unknown_commit_never_grants_pass"), and
[`test_gate_ledger.py:353-360`](../tests/test_gate_ledger.py#L353-L360 "anchor:test_contextual_pass_is_invalidated_by_package_version"),
[`test_gate_ledger.py:946-973`](../tests/test_gate_ledger.py#L946-L973 "anchor:test_contextual_authority_and_token_bind_exact_dgx_source"),
[`test_gate_ledger.py:976-1003`](../tests/test_gate_ledger.py#L976-L1003 "anchor:test_contextual_legacy_authority_without_dgx_source_reads_stale"), and
[`test_gate_ledger.py:1006-1048`](../tests/test_gate_ledger.py#L1006-L1048 "anchor:test_same_version_source_change_invalidates_after_process_cache_reset").

> **Re-verify these invariants when changing gate code.** Changes to the ledger,
> ceremony, artifact binding, slab proof, automatic gate, quarantine, Fleet
> authorization, or failed-load resource ownership must preserve all numbered
> invariants below. Update the cited implementation and regression together;
> do not infer a broader grant from a passing test for one context.

## Vocabulary and scope

- A **combination** is the checkpoint name, sorted loader-option entries, and
  sorted LoRA-name list. The production key construction is
  [`gate_ledger.py:149-152`](../src/dgx_monarch/gate_ledger.py#L149-L152 "anchor:combo_key") and
  [`gate.py:97-109`](../src/dgx_monarch/nodes/gate.py#L97-L109 "anchor:_combo_of"); ordering and
  name-change behavior are covered by
  [`test_gate_ledger.py:23-27`](../tests/test_gate_ledger.py#L23-L27 "anchor:test_combo_key_ignores_lora_order_and_detects_name_change").
- An **artifact identity** is a bounded practical fingerprint, not a whole-file
  cryptographic attestation. Each file contributes three sampled windows of up
  to 1 MiB each, at the start, middle, and end, and the aggregate binds every ordered
  signature without truncation. This is enforced by
  [`gate_artifacts.py:77-134`](../src/dgx_monarch/gate_artifacts.py#L77-L134 "anchor:artifact_signature") and
  [`gate_ledger.py:105-123`](../src/dgx_monarch/gate_ledger.py#L105-L123 "anchor:artifact_set_signature"), and
  covered by
  [`test_gate_ledger.py:48-59`](../tests/test_gate_ledger.py#L48-L59 "anchor:test_artifact_signature_tracks_bytes"),
  [`test_gate_ledger.py:62-93`](../tests/test_gate_ledger.py#L62-L93 "anchor:test_artifact_signature_bounds_io_to_sample_windows"),
  [`test_gate_ledger.py:107-129`](../tests/test_gate_ledger.py#L107-L129 "anchor:test_cached_signature_rechecks_path_after_atomic_replacement"),
  [`test_gate_ledger.py:132-136`](../tests/test_gate_ledger.py#L132-L136 "anchor:test_artifact_set_signature_binds_every_entry_without_truncation"),
  [`test_gate_ledger.py:139-144`](../tests/test_gate_ledger.py#L139-L144 "anchor:test_artifact_set_signature_is_an_explicit_copyable_value"),
  [`test_gate_ledger.py:180-188`](../tests/test_gate_ledger.py#L180-L188 "anchor:test_lookup_accepts_complete_pre_upgrade_artifact_aggregate"), and
  [`test_gate_ledger.py:191-205`](../tests/test_gate_ledger.py#L191-L205 "anchor:test_incomplete_legacy_aggregate_preserves_fail_but_not_pass").
- A **capability context** records the complete effective worker settings and
  physical/config identity used by a proof. The normal grant also binds
  topology, attention,
  and Ulysses synchronization; the separately scoped Fleet grant binds world-1
  storage behavior instead. The contexts are built at
  [`gate_identity.py:117-130`](../src/dgx_monarch/nodes/gate_identity.py#L117-L130 "anchor:resolve_effective_worker_args"),
  [`gate_identity.py:133-181`](../src/dgx_monarch/nodes/gate_identity.py#L133-L181 "anchor:gate_capability_context"), and
  [`gate_identity.py:184-202`](../src/dgx_monarch/nodes/gate_identity.py#L184-L202 "anchor:fleet_residency_capability_context") from the
  physical fields in
  [`mesh_residency.py:21-31`](../src/dgx_monarch/mesh_residency.py#L21-L31 "anchor:physical_capability_context"), and
  are covered by
  [`test_gate_orchestration.py:945-965`](../tests/test_gate_orchestration.py#L945-L965 "anchor:test_gate_context_merges_cluster_and_init_worker_args")
  and [`test_fleet.py:803-820`](../tests/test_fleet.py#L803-L820 "anchor:test_fleet_preserves_requested_policy_only_for_exact_context_pass").
  Tokens and ledger rows use the same canonical context encoder. It accepts
  only finite JSON-native values and rejects opaque metadata, non-string
  mapping keys, tuples used as arrays, and non-finite numbers. It never
  substitutes `repr()`, which may contain process addresses. The boundary is
  implemented in
  [`gate_ledger.py:52-59`](../src/dgx_monarch/gate_ledger.py#L52-L59 "anchor:_canonical_capability_context")
  and covered by
  [`test_gate_ledger.py:727-750`](../tests/test_gate_ledger.py#L727-L750 "anchor:test_capability_token_and_ledger_share_strict_canonical_encoding"),
  [`test_gate_ledger.py:753-770`](../tests/test_gate_ledger.py#L753-L770 "anchor:test_capability_context_rejects_opaque_metadata_before_grant_or_record"),
  [`test_gate_ledger.py:773-785`](../tests/test_gate_ledger.py#L773-L785 "anchor:test_capability_context_rejects_non_finite_numbers"), and
  [`test_gate_ledger.py:788-799`](../tests/test_gate_ledger.py#L788-L799 "anchor:test_capability_context_rejects_non_json_container_shapes").
- A **dgx-monarch source identity** is the cached canonical SHA-256 manifest of
  the import-capable package tree. Each new process computes it on first use;
  it is a process identity, not a live filesystem monitor. Editable installs
  and branch updates can change that identity without changing `__version__`,
  so matching protocol and package versions alone cannot authorize reuse. Every
  contextual `PASS`, `INCONCLUSIVE`, and `RETESTING` row carries `dgx_source`,
  and the same digest is part of every process verdict token. An absent or
  different digest makes the row stale and prevents process-cached results
  from authorizing it.
  Context-free diagnostics remain readable. FAIL quarantine and permanent
  audit rows remain valid across source changes because neither authorizes
  execution. Inventory
  traversal errors are re-raised, so an unreadable included subtree cannot
  drop out of a trusted manifest unnoticed. If source identity cannot be
  computed, ordinary normal/Fleet authorization mints no grant and uses stock;
  FSDP or comfy-managed paths with no stock equivalent refuse instead. The
  canonical inventory, digest, token, writer, and selector are
  [`runtime_provenance.py:101-140`](../src/dgx_monarch/runtime_provenance.py#L101-L140 "anchor:_source_inventory"),
  [`runtime_provenance.py:229-232`](../src/dgx_monarch/runtime_provenance.py#L229-L232 "anchor:cached_dgx_source_manifest_sha256"),
  [`gate_ledger.py:62-69`](../src/dgx_monarch/gate_ledger.py#L62-L69 "anchor:gate_verdict_token"),
  [`gate_ledger.py:380-407`](../src/dgx_monarch/gate_ledger.py#L380-L407 "anchor:_entry"), and
  [`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity"), covered by
  [`test_gate_ledger.py:946-973`](../tests/test_gate_ledger.py#L946-L973 "anchor:test_contextual_authority_and_token_bind_exact_dgx_source"),
  [`test_gate_ledger.py:976-1003`](../tests/test_gate_ledger.py#L976-L1003 "anchor:test_contextual_legacy_authority_without_dgx_source_reads_stale"), and
  [`test_gate_ledger.py:1006-1048`](../tests/test_gate_ledger.py#L1006-L1048 "anchor:test_same_version_source_change_invalidates_after_process_cache_reset"),
  [`test_runtime_provenance.py:132-149`](../tests/test_runtime_provenance.py#L132-L149 "anchor:test_source_inventory_rejects_unreadable_included_subtree"),
  [`test_gate_orchestration.py:3357-3375`](../tests/test_gate_orchestration.py#L3357-L3375 "anchor:test_normal_render_source_manifest_failure_forces_stock_without_grant"), and
  [`test_fleet.py:1471-1499`](../tests/test_fleet.py#L1471-L1499 "anchor:test_fleet_source_manifest_failure_forces_stock_without_a_grant").

## Ledger row lifecycle

1. `record()` converts an `ArtifactSetSignature` to its current digest, attempts
   to write the verdict and detail, and stamps every contextual row with the
   current protocol, dgx-monarch version, and serialized capability context;
   contextual `PASS` and `INCONCLUSIVE` rows also carry the canonical
   package-source manifest. `begin_retest_required()` stamps `RETESTING` with
   the protocol, version, and manifest, and lists its blocked contexts. The
   implementation is
   [`gate_ledger.py:380-407`](../src/dgx_monarch/gate_ledger.py#L380-L407 "anchor:_entry") and
   [`gate_ledger.py:478-486`](../src/dgx_monarch/gate_ledger.py#L478-L486 "anchor:record");
   row normalization and stamps are covered by
   [`test_gate_ledger.py:208-212`](../tests/test_gate_ledger.py#L208-L212 "anchor:test_record_normalizes_explicit_artifact_set_to_current_digest"),
   [`test_gate_ledger.py:247-269`](../tests/test_gate_ledger.py#L247-L269 "anchor:test_pass_is_bound_to_protocol_version_and_capability_context") and
   [`test_gate_ledger.py:946-973`](../tests/test_gate_ledger.py#L946-L973 "anchor:test_contextual_authority_and_token_bind_exact_dgx_source").
   A write `OSError` is logged. The covered pre-open failure leaves no row;
   durable reuse depends only on what a later lookup finds on disk. The
   failure path is
   [`gate_ledger.py:478-486`](../src/dgx_monarch/gate_ledger.py#L478-L486 "anchor:record"),
   covered by
   [`test_gate_ledger.py:474-487`](../tests/test_gate_ledger.py#L474-L487 "anchor:test_record_write_failure_logs_and_leaves_no_durable_row").
2. On supported Linux systems, the append attempts one advisory lock around the
   complete JSONL row. A best-effort final `record()` verdict proceeds as an
   unlocked append if `fcntl` or locking is unavailable, but the
   trust-critical `RETESTING` transaction fails closed unless it obtains the
   advisory-lock primitive. A torn prior tail is terminated before the new row,
   then the file is flushed and `fsync`ed. The write path is
   [`gate_ledger.py:409-460`](../src/dgx_monarch/gate_ledger.py#L409-L460 "anchor:_append");
   partial-record recovery, concurrent writers, and strict-lock refusal are
   covered by
   [`test_runtime_remediation.py:25-38`](../tests/test_runtime_remediation.py#L25-L38 "anchor:test_gate_ledger_ignores_only_torn_jsonl_record"),
   [`test_gate_ledger.py:452-471`](../tests/test_gate_ledger.py#L452-L471 "anchor:test_record_concurrent_writers_preserve_every_row"), and
   [`test_gate_ledger.py:625-639`](../tests/test_gate_ledger.py#L625-L639 "anchor:test_retest_guard_requires_lock_support_but_record_remains_best_effort").
3. `lookup_with_entry()` reads the ledger once and returns the state together
   with the exact row that produced it; `lookup()` only projects the state. This
   prevents a concurrent append from pairing one verdict with another row's
   quarantine levers. The single-read contract is
   [`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity")
   and [`gate_ledger.py:352-355`](../src/dgx_monarch/gate_ledger.py#L352-L355 "anchor:lookup"), and is
   covered by
   [`test_gate_ledger.py:371-387`](../tests/test_gate_ledger.py#L371-L387 "anchor:test_lookup_with_entry_pairs_state_with_its_exact_row")
   and [`test_gate_ledger.py:390-398`](../tests/test_gate_ledger.py#L390-L398 "anchor:test_lookup_delegates_to_single_read_implementation").
4. For the canonical writer outcomes, a missing combination is `unknown`. A
   known combination with no row matching its artifact and current contextual
   runtime identity, or a matching `PASS` or `INCONCLUSIVE` row with an unknown
   or changed ComfyUI commit, is `stale`. A current matching `FAIL` is `fail`;
   a current `RETESTING` guard is `inconclusive`; a known-commit `PASS` is
   `pass`; a known-commit `INCONCLUSIVE` is `inconclusive`.
   The state machine is
   [`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity") and is
   covered by
   [`test_gate_ledger.py:215-228`](../tests/test_gate_ledger.py#L215-L228 "anchor:test_lookup_semantics"),
   [`test_gate_ledger.py:346-350`](../tests/test_gate_ledger.py#L346-L350 "anchor:test_recorded_unknown_commit_never_grants_pass"),
   [`test_gate_ledger.py:353-360`](../tests/test_gate_ledger.py#L353-L360 "anchor:test_contextual_pass_is_invalidated_by_package_version"),
   [`test_gate_ledger.py:363-368`](../tests/test_gate_ledger.py#L363-L368 "anchor:test_fail_quarantine_is_sticky_across_commit_uncertainty"),
   [`test_gate_ledger.py:371-387`](../tests/test_gate_ledger.py#L371-L387 "anchor:test_lookup_with_entry_pairs_state_with_its_exact_row"), and
   [`test_gate_ledger.py:390-398`](../tests/test_gate_ledger.py#L390-L398 "anchor:test_lookup_delegates_to_single_read_implementation").

Before any ceremony re-tests existing authority, `begin_retest_required()`
appends and fsyncs one `RETESTING` row. Ordinary residency ceremonies include
the normal, Fleet, and every potential explicit-slab sibling context; a no-LoRA
FSDP ceremony includes only its separately scoped normal-render capability.
Unlike best-effort final reporting, this write is strict: failure aborts before
the first unload or proof render.
The guard denies each listed context until a newer matching verdict replaces
it. A crash or partial final-write failure therefore cannot restore an older
PASS after restart. A `RETESTING` row missing either its gate-protocol or
package-version identity is damaged evidence, not a skippable guard, and
therefore cannot resurrect older authority. An otherwise valid guard with an
absent or different package-source identity reads stale and cannot bridge
process authority. The schema, strict append, and its
placement before ceremony side effects are
[`gate_ledger.py:161-190`](../src/dgx_monarch/gate_ledger.py#L161-L190 "anchor:_valid_entry_schema"),
[`gate_ledger.py:488-525`](../src/dgx_monarch/gate_ledger.py#L488-L525 "anchor:begin_retest_required"), and
[`gate_ceremony.py:210-251`](../src/dgx_monarch/nodes/gate_ceremony.py#L210-L251 "anchor:gather_ceremony_evidence").
Atomic multi-context denial, malformed-guard refusal, strict write/lock
failure, and side-effect ordering are covered by
[`test_gate_ledger.py:490-525`](../tests/test_gate_ledger.py#L490-L525 "anchor:test_retest_guard_atomically_denies_every_context_until_exact_finals"),
[`test_gate_ledger.py:582-606`](../tests/test_gate_ledger.py#L582-L606 "anchor:test_retest_guard_missing_runtime_identity_cannot_resurrect_older_pass"),
[`test_gate_ledger.py:976-1003`](../tests/test_gate_ledger.py#L976-L1003 "anchor:test_contextual_legacy_authority_without_dgx_source_reads_stale"),
[`test_gate_ledger.py:609-622`](../tests/test_gate_ledger.py#L609-L622 "anchor:test_retest_guard_write_failure_is_not_best_effort"),
[`test_gate_ledger.py:625-639`](../tests/test_gate_ledger.py#L625-L639 "anchor:test_retest_guard_requires_lock_support_but_record_remains_best_effort"),
[`test_gate_ledger.py:1096-1113`](../tests/test_gate_ledger.py#L1096-L1113 "anchor:test_retest_source_manifest_failure_is_typed_and_writes_nothing"),
[`test_gate_orchestration.py:1612-1641`](../tests/test_gate_orchestration.py#L1612-L1641 "anchor:test_ceremony_durably_guards_every_potential_grant_before_unload"), and
[`test_gate_orchestration.py:1644-1670`](../tests/test_gate_orchestration.py#L1644-L1670 "anchor:test_required_retest_guard_failure_aborts_before_any_ceremony_leg").

Context-free rows remain readable for legacy reporting, but runtime trust
callers provide a context, so a historical context-free `PASS` reads stale and
requires a new proof. That boundary is explicit at
[`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity") and is
covered by
[`test_gate_ledger.py:247-269`](../tests/test_gate_ledger.py#L247-L269 "anchor:test_pass_is_bound_to_protocol_version_and_capability_context") and
[`test_gate_ledger.py:1116-1141`](../tests/test_gate_ledger.py#L1116-L1141 "anchor:test_source_failure_preserves_unscoped_diagnostics_and_audit_rows").

## Protocol invariants

These are the v13 invariants. v13 adds the fourth frozen class-K waiver kind,
the sharded-quant scale bar (`shard_quant_scale`). It carries the
v12 rule that every slab-cycle response binds to one exact rank, world, and
current setup generation before it can be conclusive, active, or eligible for a
no-material session skip, the
v11 sol-attn class-K waiver-vocabulary bump, the v10 repeat-identity and
package-source authority guarantees, the v9 capacity
and waiver guarantees, the v8 FSDP clean-reload guarantees, and the v7
packed-latent guarantees, all re-verified against the current tree rather than
copied from historical evidence.

1. **`GATE_PROTOCOL_VERSION == 13`; v4 through v12 are burned forever.** v4
   records rejected `slab_unreferenced` PASS authority and bidirectional
   `auto`/`on` sibling stamping. v5 lacks the deep immutable transaction,
   exact normal-render grant, and crash-durable multi-context retest boundary.
   Two divergent v6 lineages exist: a packed-comparison prototype adds finite
   per-modality identity without the later transaction and grant hardening,
   while the deployed v6 has that hardening but keeps the
   top-level comparison and skips packed modalities in its immutable
   transaction walker. v7 is the first protocol to combine both boundaries,
   but it gives resolved no-LoRA FSDP no dedicated proof that each rank
   independently unloads, loads, and freshly shards the exact artifact without
   slab residency. v8 adds that exact
   `fsdp_clean_reload_v1` authority scope, so no older protocol can authorize
   current residency or FSDP capacity execution. v8 predates the rule
   that a pre-gate slab load is permitted only under a byte-verify certificate,
   and predates the permanent waiver row every consented bypass writes. v9
   introduces those requirements, but its no-swap cross-mode branch can grant
   from the stock-versus-B comparison even when the same-residency A/B repeat
   diverges. v10 requires an exact A/B repeat for every PASS; a passing cross
   comparison with a divergent repeat is INCONCLUSIVE. It also adds the
   package version and first-use-cached canonical package-source manifest to
   contextual rows, selectors, and execution tokens, and requires the exact
   source-matched all-rank proof cohort before and after a ceremony. v11 adds
   the sol-attn kind to the frozen waiver vocabulary. v12 rejects a slab
   proof that only has the right number of responses: every row must be an
   exact dict for a unique rank in `0..world-1`, report that same world, and
   report the handle's current positive setup generation. A malformed,
   duplicate, missing, out-of-range, wrong-world, or stale-generation row
   grants neither a conclusive/active proof nor a no-material skip. v13 adds
   a fourth class-K waiver kind, so a row written under the three-kind
   vocabulary cannot stand as current. Every
   v12-and-older PASS therefore re-proves once per (combination, artifact digest,
   comfy commit, capability context, Gate protocol, package version,
   package-source manifest) on first use. There is no migration and no
   backfill: the JSONL keeps every older row, and older rows stay stale.
   The burn and selector are
   [`gate_ledger.py:49-49`](../src/dgx_monarch/gate_ledger.py#L49-L49 "anchor:GATE_PROTOCOL_VERSION") and
   [`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity"); stale v5 and v6
   authority, both rejected v4 shapes, and the v7-to-v8, v8-to-v9, v9-to-v10,
   v11-to-v12, and v12-to-v13 burns are covered by
   [`test_gate_ledger.py:272-288`](../tests/test_gate_ledger.py#L272-L288 "anchor:test_v5_pass_without_frozen_transaction_inputs_reads_stale") and
   [`test_gate_ledger.py:291-308`](../tests/test_gate_ledger.py#L291-L308 "anchor:test_divergent_v6_lineages_read_stale_under_current_protocol"),
   [`test_gate_ledger.py:802-830`](../tests/test_gate_ledger.py#L802-L830 "anchor:test_v7_pass_without_fsdp_clean_reload_reads_stale_under_v8"),
   [`test_gate_ledger.py:833-860`](../tests/test_gate_ledger.py#L833-L860 "anchor:test_v8_pass_without_the_byte_verify_rule_reads_stale_under_v9"),
   [`test_gate_ledger.py:919-943`](../tests/test_gate_ledger.py#L919-L943 "anchor:test_v9_cross_mode_pass_without_repeat_identity_reads_stale_under_v10"),
   [`test_gate_ledger.py:311-343`](../tests/test_gate_ledger.py#L311-L343 "anchor:test_burned_v4_rows_read_stale_not_current_trust"),
   [`test_gate_ledger.py:863-887`](../tests/test_gate_ledger.py#L863-L887 "anchor:test_v11_pass_reads_stale_under_v12"), and
   [`test_gate_ledger.py:891-916`](../tests/test_gate_ledger.py#L891-L916 "anchor:test_v12_pass_reads_stale_under_v13").
2. **PASS binds to capability context, protocol version, package version, and
   canonical package-source manifest; any change re-proves.** Contextual rows
   are stamped and selected at
   [`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity") and
   [`gate_ledger.py:478-486`](../src/dgx_monarch/gate_ledger.py#L478-L486 "anchor:record").
   One ceremony also resolves and claims one handle before entering the gate
   engine, freezes the complete effective worker policy into a clone pinned to
   that handle, and uses that clone for every render and capability context.
   Before verdict construction can reach `PASS`, the engine compares the live
   graph policy with the snapshot. Any drift aborts and records no grant. An
   ordinary residency ceremony forces the mutable graph back to stock; an FSDP
   ceremony instead unloads its exact handle or leaves that handle DIRTY or
   defunct because no stock-residency equivalent exists in the same topology.
   The claim,
   clone, and pre-verdict drift check are enforced at
   [`gate_session.py:19-169`](../src/dgx_monarch/nodes/gate_session.py#L19-L169 "anchor:run_identity_ceremony"),
   [`gate_identity.py:259-274`](../src/dgx_monarch/nodes/gate_identity.py#L259-L274 "anchor:model_with_worker_overrides"), and
   [`gate_ceremony.py:100-356`](../src/dgx_monarch/nodes/gate_ceremony.py#L100-L356 "anchor:gather_ceremony_evidence"); policy
   drift is covered by
   [`test_gate_orchestration.py:2016-2047`](../tests/test_gate_orchestration.py#L2016-L2047 "anchor:test_identity_ceremony_rejects_policy_drift_before_pass").
   The immutable transaction walker accepts only the direct class exported
   by `comfy.nested_tensor`. It visits every modality without looping on
   cycles, owns one cloned baseline, and shares the guarded tensor leaves
   across separate containers for each proof render. Each leaf carries both
   its PyTorch version counter and a chunked
   full-storage SHA-256 fingerprint, so `.data`, raw-storage, NumPy-view, and
   zero-length storage-replacement writes cannot bypass the guard. A final
   check immediately before verdict derivation also closes retained-reference
   mutation after a render's post-call check. Mutation of either modality
   aborts without recording a grant. The capture and content guard are
   [`gate_identity.py:39-76`](../src/dgx_monarch/nodes/gate_identity.py#L39-L76 "anchor:_transaction_tensor_memo"),
   [`gate_identity.py:79-81`](../src/dgx_monarch/nodes/gate_identity.py#L79-L81 "anchor:freeze_transaction"), and
   [`gate_identity.py:89-100`](../src/dgx_monarch/nodes/gate_identity.py#L89-L100 "anchor:transaction_tensor_versions") and
   [`gate_identity.py:103-114`](../src/dgx_monarch/nodes/gate_identity.py#L103-L114 "anchor:require_transaction_unchanged"), with bounded-scratch full-storage hashing at
   [`gate_tensor_guard.py:71-86`](../src/dgx_monarch/nodes/gate_tensor_guard.py#L71-L86 "anchor:transaction_tensor_fingerprint"), covered by
   [`test_gate_orchestration.py:498-529`](../tests/test_gate_orchestration.py#L498-L529 "anchor:test_transaction_snapshot_clones_and_shares_every_packed_modality"),
   [`test_gate_orchestration.py:532-549`](../tests/test_gate_orchestration.py#L532-L549 "anchor:test_transaction_guard_rejects_mutation_of_either_packed_modality"), and
   [`test_gate_orchestration.py:564-583`](../tests/test_gate_orchestration.py#L564-L583 "anchor:test_transaction_guard_rejects_unversioned_packed_modality_mutation"),
   [`test_gate_orchestration.py:602-625`](../tests/test_gate_orchestration.py#L602-L625 "anchor:test_transaction_snapshot_and_fingerprint_reject_tensor_subclasses"),
   [`test_gate_orchestration.py:662-680`](../tests/test_gate_orchestration.py#L662-L680 "anchor:test_transaction_guard_rejects_zero_numel_storage_replacement"), and
   [`test_gate_orchestration.py:1925-1967`](../tests/test_gate_orchestration.py#L1925-L1967 "anchor:test_ceremony_rechecks_a_retained_packed_reference_before_grant").
   Context, protocol, package, and same-version source invalidation are covered by
   [`test_gate_ledger.py:247-269`](../tests/test_gate_ledger.py#L247-L269 "anchor:test_pass_is_bound_to_protocol_version_and_capability_context"),
   [`test_gate_ledger.py:272-288`](../tests/test_gate_ledger.py#L272-L288 "anchor:test_v5_pass_without_frozen_transaction_inputs_reads_stale"),
   [`test_gate_ledger.py:311-343`](../tests/test_gate_ledger.py#L311-L343 "anchor:test_burned_v4_rows_read_stale_not_current_trust"), and
   [`test_gate_ledger.py:353-360`](../tests/test_gate_ledger.py#L353-L360 "anchor:test_contextual_pass_is_invalidated_by_package_version"),
   [`test_gate_ledger.py:946-973`](../tests/test_gate_ledger.py#L946-L973 "anchor:test_contextual_authority_and_token_bind_exact_dgx_source"),
   [`test_gate_ledger.py:976-1003`](../tests/test_gate_ledger.py#L976-L1003 "anchor:test_contextual_legacy_authority_without_dgx_source_reads_stale"), and
   [`test_gate_ledger.py:1006-1048`](../tests/test_gate_ledger.py#L1006-L1048 "anchor:test_same_version_source_change_invalidates_after_process_cache_reset").
3. **Same-context FAIL is sticky; automatic gating does not retry it into PASS
   without an identity change.** Stickiness is scoped to the same combination,
   artifact, current protocol/package/context, and survives both ComfyUI commit
   uncertainty and package-source drift; source identity constrains grants, not
   quarantine. An explicit ceremony may supersede it with a newer row. Lookup
   enforces this at
   [`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity"), and
   the automatic path accepts `fail` as terminal at
   [`auto_gate.py:306-505`](../src/dgx_monarch/nodes/auto_gate.py#L306-L505 "anchor:maybe_auto_gate").
   Coverage is
   [`test_gate_ledger.py:363-368`](../tests/test_gate_ledger.py#L363-L368 "anchor:test_fail_quarantine_is_sticky_across_commit_uncertainty"),
   [`test_gate_ledger.py:1006-1048`](../tests/test_gate_ledger.py#L1006-L1048 "anchor:test_same_version_source_change_invalidates_after_process_cache_reset"),
   [`test_gate_ledger.py:1053-1087`](../tests/test_gate_ledger.py#L1053-L1087 "anchor:test_foreign_source_row_cannot_displace_sticky_fail_or_revive_older_pass"), and
   [`test_gate_orchestration.py:1167-1191`](../tests/test_gate_orchestration.py#L1167-L1191 "anchor:test_persisted_fail_quarantines_fresh_model_before_setup").
4. **`auto` to explicit `on` stamping is one-directional.** A sibling
   grant is emitted only for `PASS`, all-rank active slab residency, a passing
   slab-vs-stock leg, and a vouched family. Explicit `on` never stamps the
   hardware-dependent omitted/auto sibling. Revocation is bounded by the same
   evidence: a `FAIL` reaches the sibling rows only when the cross-residency
   leg itself diverged, because that leg is the only part of a ceremony slab
   residency runs. A swap-lineage `FAIL` with no failing cross leg writes no
   sibling row at all, so the untested explicit-`on` capability keeps the
   preflight `RETESTING` denial and re-gates on next use instead of taking a
   sticky `FAIL` it never earned. An `INCONCLUSIVE` that found nothing to gate
   revokes the sibling rows in the ledger only. The process cache denies the
   one context that ceremony exercised; the sibling contexts lose the retest
   guard's denial the ceremony opened on them and any session `PASS` the row
   revoked, and gain no denial. The first explicit-`on` or Fleet render of the
   combination then runs its own ceremony instead of taking a quarantine from a
   ceremony that never ran slab residency (see
   [`gate_inconclusive.py:211-225`](../src/dgx_monarch/nodes/gate_inconclusive.py#L211-L225 "anchor:session_denial_tokens")). A sibling `FAIL` row written before the cross-leg rule
   retires with the protocol and package binding above;
   [docs/TROUBLESHOOTING.md #81](TROUBLESHOOTING.md#81-explicit-slab-residency-still-reads-quarantined-after-a-lora-only-gate-fail) carries the operator recovery. Coverage for the
   cross-leg boundary is
   [`test_gate_orchestration.py:2071-2090`](../tests/test_gate_orchestration.py#L2071-L2090 "anchor:test_cross_residency_divergence_under_auto_revokes_the_explicit_on_sibling") and
   [`test_gate_orchestration.py:2093-2120`](../tests/test_gate_orchestration.py#L2093-L2120 "anchor:test_lazy_swap_failure_under_auto_leaves_the_explicit_on_sibling_untested"). The conditions are
   [`slab_proof.py:23-90`](../src/dgx_monarch/nodes/slab_proof.py#L23-L90 "anchor:_cycle_state") and
   [`gate_identity.py:518-577`](../src/dgx_monarch/nodes/gate_identity.py#L518-L577 "anchor:equivalent_slab_mode_contexts")
   and the rows are written at
   [`gate_verdict.py:205-248`](../src/dgx_monarch/nodes/gate_verdict.py#L205-L248 "anchor:publish_ceremony_verdict"). Positive,
   reverse, config-supplied-auto, and unvouched cases are covered by
   [`test_gate_orchestration.py:1673-1704`](../tests/test_gate_orchestration.py#L1673-L1704 "anchor:test_cross_mode_reference_passes_and_restores_slab")
   and
   [`test_gate_orchestration.py:2628-2642`](../tests/test_gate_orchestration.py#L2628-L2642 "anchor:test_pass_under_auto_stamps_the_explicit_on_context"),
   [`test_gate_orchestration.py:2645-2660`](../tests/test_gate_orchestration.py#L2645-L2660 "anchor:test_config_supplied_auto_stamps_only_the_explicit_on_variant"), and
   [`test_gate_orchestration.py:2663-2674`](../tests/test_gate_orchestration.py#L2663-L2674 "anchor:test_pass_with_unvouched_family_never_stamps"),
   with partial-rank refusal covered by
   [`test_gate_orchestration.py:3083-3117`](../tests/test_gate_orchestration.py#L3083-L3117 "anchor:test_explicit_slab_without_all_rank_residency_never_passes_or_crosses").
5. **CAPACITY never grants; generic OOM in the cross-residency reference leg
   remains ERROR then INCONCLUSIVE.** Only a typed capacity refusal is
   classified as `CAPACITY`: a `StockLoadCapacityError`, raised at a load or
   activation boundary, or a subclass of it such as the driver-side footprint
   refusal. The classifier matches it by isinstance, and by class name once
   Monarch wraps it. An errored or capacity-limited reference makes the verdict
   `INCONCLUSIVE`, even if the other comparisons pass. A capacity-skipped cross
   leg may also
   record a `CAPACITY_CERTIFIED` audit row carrying the byte-verify certificate
   summary for the slab side of the comparison. **That row grants nothing.** It
   attests only that the slab contains the checkpoint's bytes. The stock
   reference did not run, so pixel fidelity is unproven and the verdict stays
   `INCONCLUSIVE`. Neither a certificate nor a consent
   can convert an INCONCLUSIVE into a PASS.
   Classification and verdict logic are
   [`mesh_safety.py:255-256`](../src/dgx_monarch/mesh_safety.py#L255-L256 "anchor:StockLoadCapacityError"),
   [`mesh_safety.py:259-263`](../src/dgx_monarch/mesh_safety.py#L259-L263 "anchor:is_stock_load_capacity_error"),
   [`driver_footprint.py:103-103`](../src/dgx_monarch/driver_footprint.py#L103-L103 "anchor:DriverFootprintCapacityError"), and
   [`gate_verdict.py:73-95`](../src/dgx_monarch/nodes/gate_verdict.py#L73-L95 "anchor:publish_ceremony_verdict"). A driver-side
   refusal raised inside a first-use leg is covered by
   [`test_driver_footprint_preflight.py:530-579`](../tests/test_driver_footprint_preflight.py#L530-L579 "anchor:test_a_gate_leg_reports_it_as_capacity_with_the_numbers_intact").
   Typed, wrapped, and generic-OOM cases are covered by
   [`test_gate_orchestration.py:2677-2696`](../tests/test_gate_orchestration.py#L2677-L2696 "anchor:test_capacity_refusal_stays_inconclusive_with_no_pass_grant"),
   [`test_gate_orchestration.py:2699-2711`](../tests/test_gate_orchestration.py#L2699-L2711 "anchor:test_wrapped_capacity_refusal_is_classified_but_grants_nothing"),
   [`test_gate_orchestration.py:2714-2727`](../tests/test_gate_orchestration.py#L2714-L2727 "anchor:test_generic_oom_in_reference_render_stays_inconclusive"),
   [`test_gate_orchestration.py:2730-2749`](../tests/test_gate_orchestration.py#L2730-L2749 "anchor:test_oom_during_stock_policy_apply_stays_inconclusive"), and
   [`test_gate_orchestration.py:2752-2763`](../tests/test_gate_orchestration.py#L2752-L2763 "anchor:test_oom_cross_leg_under_auto_stays_inconclusive").
6. **The cross-residency leg runs whenever worker reports prove coherent,
   policy-expected all-rank slab residency, including a vouched auto default.**
   The proof derives all-rank active state from one exact current cycle cohort,
   not driver prediction: every response must be a unique rank in the handle's
   world and bind that world plus the current setup generation. It refuses
   malformed, duplicate, stale, unexpected, unvouched, or partial residency before
   comparison at
   [`slab_proof.py:23-90`](../src/dgx_monarch/nodes/slab_proof.py#L23-L90 "anchor:_cycle_state") and
   [`slab_proof.py:93-173`](../src/dgx_monarch/nodes/slab_proof.py#L93-L173 "anchor:establish").
   For a valid proof, every worker must then leave slab before a fresh stock
   load at [`gate_ceremony.py:234-256`](../src/dgx_monarch/nodes/gate_ceremony.py#L234-L256 "anchor:gather_ceremony_evidence").
   FSDP disables slab residency, and it takes a LoRA stack only with
   `lora_low_rss` on at
   [`fsdp.py:353-365`](../src/dgx_monarch/adapters/fsdp.py#L353-L365 "anchor:validate_fsdp_launch_loras"),
   so a no-LoRA FSDP ceremony has a separate exact proof rather than counting
   either risky lineage as run.
   Under a resolved FSDP capability only, every worker must hold the bound
   artifact identity, unload, independently load and shard the model, report a
   fresh non-slab `load`, and then reproduce the first clean-load output
   exactly. Missing identity, partial-rank evidence, reuse, unexpected slab
   residency, or generic no-LoRA worker metadata grants nothing. A divergence
   records FAIL for the exact FSDP context without misdiagnosing or toggling a
   LoRA/slab quarantine lever. The worker cycle, complete-rank proof, and
   verdict binding are
   [`gate_fsdp_cycle.py:59-193`](../src/dgx_monarch/actor/gate_fsdp_cycle.py#L59-L193 "anchor:run"),
   [`gate_fsdp.py:305-469`](../src/dgx_monarch/nodes/gate_fsdp.py#L305-L469 "anchor:establish"), and
   [`gate_verdict.py:131-227`](../src/dgx_monarch/nodes/gate_verdict.py#L131-L227 "anchor:publish_ceremony_verdict"), with coverage in
   [`test_gate_fsdp_cycle_unit.py:61-87`](../tests/test_gate_fsdp_cycle_unit.py#L61-L87 "anchor:test_fsdp_reload_cycle_proves_one_fresh_ready_non_slab_load"),
   [`test_gate_fsdp_proof.py:63-67`](../tests/test_gate_fsdp_proof.py#L63-L67 "anchor:test_fsdp_reload_proof_accepts_complete_exact_rank_evidence"), and
   [`test_gate_orchestration.py:2133-2164`](../tests/test_gate_orchestration.py#L2133-L2164 "anchor:test_fsdp_no_lora_clean_reload_can_earn_exact_capability_pass").
   Audited family-specific mixed-live classification and public precision
   admission remain adapter-owned; actor and gate code consume only their
   generic evidence contract at
   [`fsdp.py:166-166`](../src/dgx_monarch/adapters/fsdp.py#L166-L166 "anchor:classify_audited_live_precision") and
   [`fsdp.py:188-188`](../src/dgx_monarch/adapters/fsdp.py#L188-L188 "anchor:fsdp_precision_profile_is_admitted").
   Each FSDP injection runs only cheap quant and LoRA refusals before adapter
   and compile preparation. The sole authoritative live-precision scan runs
   after that preparation and immediately before the first FSDP parameter
   mutation; direct calls use the same path. This is enforced by
   [`store_fsdp.py:100-117`](../src/dgx_monarch/actor/store_fsdp.py#L100-L117 "anchor:validate_injection") and
   [`fsdp.py:377-435`](../src/dgx_monarch/adapters/fsdp.py#L377-L435 "anchor:apply_fsdp_capacity_mode"), with one-scan and same-patcher drift coverage in
   [`test_fsdp_capacity.py:1095-1095`](../tests/test_fsdp_capacity.py#L1095-L1095 "anchor:test_worker_runs_one_live_precision_scan_after_preparation"),
   [`test_fsdp_capacity.py:1125-1125`](../tests/test_fsdp_capacity.py#L1125-L1125 "anchor:test_worker_live_scan_observes_same_patcher_model_replacement"), and
   [`test_fsdp_capacity.py:1166-1166`](../tests/test_fsdp_capacity.py#L1166-L1166 "anchor:test_worker_live_scan_observes_dtype_mutation_during_compile_preparation").
   Normal stock/swap and slab/fresh comparisons route through the same
   fail-closed comparator at
   [`latent_identity.py:58-118`](../src/dgx_monarch/nodes/latent_identity.py#L58-L118 "anchor:compare_latents").
   A passing slab-versus-stock cross comparison cannot replace the same-residency
   repeat: v10 permits PASS only when A and B are also exactly identical. If the
   cross comparison passes but A/B diverges, the ceremony is INCONCLUSIVE and
   publishes only INCONCLUSIVE capability verdicts; an auto/vouched slab
   ceremony likewise stamps no sibling PASS authority. The explicit and
   auto/vouched production regressions are
   [`test_gate_orchestration.py:3318-3354`](../tests/test_gate_orchestration.py#L3318-L3354 "anchor:test_no_lora_cross_pass_cannot_override_explicit_slab_repeat_divergence") and
   [`test_gate_orchestration.py:3378-3415`](../tests/test_gate_orchestration.py#L3378-L3415 "anchor:test_no_lora_cross_pass_cannot_stamp_auto_vouched_slab_sibling_passes").
   A packed result is PASS-eligible only when both values use the exact direct
   ComfyUI `NestedTensor` class, expose the same nonempty modality count, and
   every leaf is an exact `torch.Tensor` that agrees in full shape, dtype,
   layout, device, finite status, and exact value. A wrapper or tensor
   subclass, same-named spoof, malformed wrapper, mixed representation,
   non-finite value, or structural mismatch grants nothing.
   Direct-wrapper parsing is
   [`latent_identity.py:7-55`](../src/dgx_monarch/nodes/latent_identity.py#L7-L55 "anchor:direct_nested_tensor_parts"), with component and type coverage in
   [`test_latent_identity.py:70-81`](../tests/test_latent_identity.py#L70-L81 "anchor:test_tensor_subclass_cannot_override_flat_or_packed_identity"),
   [`test_latent_identity.py:84-96`](../tests/test_latent_identity.py#L84-L96 "anchor:test_nested_latents_compare_every_modality_and_aggregate_diff"),
   [`test_latent_identity.py:112-126`](../tests/test_latent_identity.py#L112-L126 "anchor:test_nested_comparison_requires_the_exact_direct_wrapper_type"), and
   [`test_latent_identity.py:209-220`](../tests/test_latent_identity.py#L209-L220 "anchor:test_nonfinite_plain_or_nested_latents_never_grant_identity").
   Production normal, real-wrapper, cross-mode, and singleton non-finite regressions are
   [`test_gate_orchestration.py:1707-1726`](../tests/test_gate_orchestration.py#L1707-L1726 "anchor:test_nested_normal_ceremony_passes_and_serializes_without_latents"),
   [`test_gate_orchestration.py:1729-1750`](../tests/test_gate_orchestration.py#L1729-L1750 "anchor:test_real_comfy_nested_tensor_runs_identity_ceremony"),
   [`test_gate_orchestration.py:1852-1876`](../tests/test_gate_orchestration.py#L1852-L1876 "anchor:test_nested_slab_cross_mode_divergence_returns_stock_and_quarantines_slab"), and
   [`test_gate_orchestration.py:1970-1989`](../tests/test_gate_orchestration.py#L1970-L1989 "anchor:test_world1_nonfinite_nested_ceremony_fails_and_quarantines").
   The actor and driver apply the same exact-leaf rule at
   [`latent_outputs.py:61-94`](../src/dgx_monarch/actor/latent_outputs.py#L61-L94 "anchor:direct_nested_modalities") and
   [`latent_outputs.py:62-84`](../src/dgx_monarch/nodes/latent_outputs.py#L62-L84 "anchor:_validate_return_tensor"). Driver-side primary/denoised independence
   compares complete nonempty storage byte intervals, so shifted overlapping
   views cannot hide behind distinct storage objects; zero-length
   shared-storage detection uses public storage-object identity. The boundary is
   [`latent_outputs.py:115-135`](../src/dgx_monarch/nodes/latent_outputs.py#L115-L135 "anchor:_shares_storage"), covered for flat and packed outputs by
   [`test_adapter_remediation.py:876-905`](../tests/test_adapter_remediation.py#L876-L905 "anchor:test_shifted_distinct_storages_are_detected_as_aliases"),
   [`test_adapter_remediation.py:908-945`](../tests/test_adapter_remediation.py#L908-L945 "anchor:test_packed_empty_views_detect_shared_storage_without_false_positive"),
   [`test_sampling_dp.py:554-580`](../tests/test_sampling_dp.py#L554-L580 "anchor:test_sampler_leader_paths_reject_tensor_subclass_outputs"), and
   [`test_pipeline.py:975-985`](../tests/test_pipeline.py#L975-L985 "anchor:test_pipeline_rejects_tensor_subclass_outputs").
   Auto-active, policy-mismatch, failed-flip, fresh-load-ordering, and divergence
   cases are covered by
   [`test_gate_orchestration.py:2050-2068`](../tests/test_gate_orchestration.py#L2050-L2068 "anchor:test_cross_mode_divergence_fails_and_quarantines_slab")
   and
   [`test_gate_orchestration.py:2766-2779`](../tests/test_gate_orchestration.py#L2766-L2779 "anchor:test_cross_mode_refuses_a_worker_that_stayed_in_slab"),
   [`test_gate_orchestration.py:2782-2793`](../tests/test_gate_orchestration.py#L2782-L2793 "anchor:test_cross_mode_unloads_before_the_stock_reference"),
   [`test_gate_orchestration.py:2796-2821`](../tests/test_gate_orchestration.py#L2796-L2821 "anchor:test_cross_mode_fail_quarantines_slab_even_when_swaps_inconclusive"),
   [`test_gate_orchestration.py:2824-2833`](../tests/test_gate_orchestration.py#L2824-L2833 "anchor:test_both_levers_quarantine_when_both_findings_hold"), and
   [`test_gate_orchestration.py:2836-2860`](../tests/test_gate_orchestration.py#L2836-L2860 "anchor:test_cross_mode_triggers_on_worker_reported_residency"),
   plus the policy refusals at
   [`test_gate_orchestration.py:2981-3003`](../tests/test_gate_orchestration.py#L2981-L3003 "anchor:test_auto_unvouched_active_slab_is_a_policy_error"),
   [`test_gate_orchestration.py:3006-3033`](../tests/test_gate_orchestration.py#L3006-L3033 "anchor:test_auto_partial_unvouched_slab_cannot_pass"), and
   [`test_gate_orchestration.py:3036-3056`](../tests/test_gate_orchestration.py#L3036-L3056 "anchor:test_explicit_stock_context_refuses_unexpected_active_slab").
7. **A live ceremony diagnoses implicated levers; persisted FAIL always returns
   to the complete stock residency policy.** A slab-vs-stock divergence
   disables `slab_weights`; an in-mode lazy-swap
   divergence disables `lora_low_rss`; both findings disable both. Lever
   selection and attempted row detail are
   [`gate_verdict.py:166-238`](../src/dgx_monarch/nodes/gate_verdict.py#L166-L238 "anchor:publish_ceremony_verdict"). A later
   session reads the verdict and exact row together before setup, uses the
   lever list only as diagnosis, and disables both `slab_weights` and
   `lora_low_rss`, plus `comfy_managed` when the graph carries it, at
   [`render_quarantine.py:49-124`](../src/dgx_monarch/nodes/render_quarantine.py#L49-L124 "anchor:_enforce_persisted_quarantine").
   Lever selection, successful row content, and exact-row recovery are covered by
   [`test_gate_orchestration.py:307-395`](../tests/test_gate_orchestration.py#L307-L395 "anchor:test_canonical_failure_quarantines_and_persists"),
   [`test_gate_orchestration.py:1167-1191`](../tests/test_gate_orchestration.py#L1167-L1191 "anchor:test_persisted_fail_quarantines_fresh_model_before_setup"),
   [`test_gate_orchestration.py:1194-1212`](../tests/test_gate_orchestration.py#L1194-L1212 "anchor:test_persisted_fail_lever_metadata_is_diagnostic_and_fails_closed"),
   [`test_gate_orchestration.py:1261-1290`](../tests/test_gate_orchestration.py#L1261-L1290 "anchor:test_persisted_quarantine_lookup_error_forces_both_paths_off"),
   [`test_gate_orchestration.py:1293-1318`](../tests/test_gate_orchestration.py#L1293-L1318 "anchor:test_persisted_fail_forces_all_risky_levers_off_via_real_ledger"),
   [`test_gate_orchestration.py:1321-1343`](../tests/test_gate_orchestration.py#L1321-L1343 "anchor:test_persisted_quarantine_skips_when_fail_was_superseded"),
   [`test_gate_orchestration.py:2796-2821`](../tests/test_gate_orchestration.py#L2796-L2821 "anchor:test_cross_mode_fail_quarantines_slab_even_when_swaps_inconclusive"),
   and [`test_gate_orchestration.py:2824-2833`](../tests/test_gate_orchestration.py#L2824-L2833 "anchor:test_both_levers_quarantine_when_both_findings_hold").
8. **Fleet is fail-closed.** Fleet bypasses normal `run_render` gating, so only
   an exact separately scoped world-1 `pass` carries a risky-residency grant;
   every other state, lookup error, and source-manifest failure forces `slab_weights=False` and
   `lora_low_rss=False` for that call. Driver authorization is
   [`fleet_policy.py:25-149`](../src/dgx_monarch/nodes/fleet_policy.py#L25-L149 "anchor:_fleet_worker_policy"). The grant
   carries the exact source-bound Gate token. It is validated against context,
   world, policy, full requests, artifacts, selected-worker parity, and finally
   recomputed from the worker's source identity before load and adoption at
   [`mesh_residency.py:154-234`](../src/dgx_monarch/mesh_residency.py#L154-L234 "anchor:assert_fleet_residency_grant"),
   [`mesh_rpc.py:225-340`](../src/dgx_monarch/mesh_rpc.py#L225-L340 "anchor:verify_request_artifacts"),
   [`worker_authorization.py:5-52`](../src/dgx_monarch/actor/worker_authorization.py#L5-L52 "anchor:verify_sample_artifact_authorization"), and
   [`worker_authorization.py:96-106`](../src/dgx_monarch/actor/worker_authorization.py#L96-L106 "anchor:assert_resident_artifact_identity"),
   with production call order at
   [`sample_protocol.py:150-395`](../src/dgx_monarch/actor/sample_protocol.py#L150-L395 "anchor:run_sample").
   Driver and worker failures are covered by
   [`test_fleet.py:677-711`](../tests/test_fleet.py#L677-L711 "anchor:test_fleet_forces_stock_policy_until_exact_context_pass"),
   [`test_fleet.py:752-820`](../tests/test_fleet.py#L752-L820 "anchor:test_fleet_preserves_requested_policy_only_for_exact_context_pass"),
   [`test_fleet.py:823-845`](../tests/test_fleet.py#L823-L845 "anchor:test_fleet_omitted_slab_default_and_lookup_error_both_fail_closed"),
   [`test_fleet.py:848-948`](../tests/test_fleet.py#L848-L948 "anchor:test_fleet_dispatch_binds_authorized_policy_before_setup_and_submit"),
   [`test_runtime_remediation.py:423-460`](../tests/test_runtime_remediation.py#L423-L460 "anchor:test_direct_fleet_submit_marks_and_rejects_risky_unconditional_model"),
   [`test_runtime_remediation.py:463-513`](../tests/test_runtime_remediation.py#L463-L513 "anchor:test_fleet_grant_binds_full_unconditional_model_snapshot"),
   [`test_runtime_remediation.py:576-604`](../tests/test_runtime_remediation.py#L576-L604 "anchor:test_gate_swap_cycle_rejects_replacement_between_transitions"),
   [`test_runtime_remediation.py:607-645`](../tests/test_runtime_remediation.py#L607-L645 "anchor:test_fleet_replacement_between_authorization_and_submit_is_rejected"),
   [`test_runtime_remediation.py:648-688`](../tests/test_runtime_remediation.py#L648-L688 "anchor:test_fleet_grant_still_requires_selected_worker_parity"),
   [`test_runtime_remediation.py:691-720`](../tests/test_runtime_remediation.py#L691-L720 "anchor:test_worker_rechecks_fleet_grant_after_driver_preflight"),
   [`test_runtime_remediation.py:723-760`](../tests/test_runtime_remediation.py#L723-L760 "anchor:test_worker_binds_loaded_model_to_grant_after_execution_recheck"),
   [`test_fleet.py:1471-1499`](../tests/test_fleet.py#L1471-L1499 "anchor:test_fleet_source_manifest_failure_forces_stock_without_a_grant"), and
   [`test_runtime_remediation.py:1672-1709`](../tests/test_runtime_remediation.py#L1672-L1709 "anchor:test_fleet_grant_rejects_driver_worker_source_manifest_mismatch").
9. **`auto_gate=first_use` gates normal KSampler and KSampler Advanced paths
   before the user's render, and LoRA-less slab renders still gate.** Omitted
   `slab_weights` remains risky even without a LoRA stack, and the sequential
   and pipeline paths prove or quarantine before submission. Risk selection and
   ordering are
   [`auto_gate.py:106-224`](../src/dgx_monarch/nodes/auto_gate.py#L106-L224 "anchor:auto_gate_context"),
   [`auto_gate.py:302-501`](../src/dgx_monarch/nodes/auto_gate.py#L302-L501 "anchor:maybe_auto_gate"), and
   [`pipeline.py:282-356`](../src/dgx_monarch/nodes/pipeline.py#L282-L356 "anchor:_push_bound").
   LoRA-less slab gating and submit ordering are covered by
   [`test_gate_orchestration.py:1001-1048`](../tests/test_gate_orchestration.py#L1001-L1048 "anchor:test_low_rss_off_does_not_suppress_slab_gate"),
   [`test_gate_orchestration.py:898-922`](../tests/test_gate_orchestration.py#L898-L922 "anchor:test_full_render_is_submitted_only_after_unproven_paths_are_quarantined"),
   [`test_pipeline.py:427-453`](../tests/test_pipeline.py#L427-L453 "anchor:test_pipeline_gates_before_first_unproven_submit_and_quarantines"), and the
   packed end-to-end first-use path at
   [`test_gate_orchestration.py:1753-1805`](../tests/test_gate_orchestration.py#L1753-L1805 "anchor:test_first_use_auto_gate_runs_production_nested_ceremony").
10. **The ledger resists concurrent writers, partial writes, and corrupt-row
    authority resurrection.** A `record()` append takes the advisory lock when
    it can, and a `RETESTING` append refuses to proceed without it. Every
    append repairs a torn tail and `fsync`s, as ledger row lifecycle step 2
    describes.
    Readers validate decoded-row schema and retain line positions: parseable
    negative rows remain denials, damage newer than an exact PASS yields
    `error`, and a later valid exact row heals older damage. Damage with no
    intact matching identity preserves the canonical `unknown`/`stale` state
    needed to re-prove that identity, but the same single-scan result marks a
    process-local PASS unsafe until healing. Existing ledgers that cannot be
    read raise a typed error rather than looking empty. The paths are
    [`gate_ledger.py:161-190`](../src/dgx_monarch/gate_ledger.py#L161-L190 "anchor:_valid_entry_schema"),
    [`gate_ledger.py:192-234`](../src/dgx_monarch/gate_ledger.py#L192-L234 "anchor:entries_with_integrity"),
    [`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity"), and
    [`gate_ledger.py:409-460`](../src/dgx_monarch/gate_ledger.py#L409-L460 "anchor:_append").
    Torn-tail, 16-writer, unreadable-ledger, positional-healing, and schema
    regressions are
    [`test_runtime_remediation.py:25-38`](../tests/test_runtime_remediation.py#L25-L38 "anchor:test_gate_ledger_ignores_only_torn_jsonl_record"),
    [`test_gate_ledger.py:452-471`](../tests/test_gate_ledger.py#L452-L471 "anchor:test_record_concurrent_writers_preserve_every_row"),
    [`test_gate_ledger.py:642-654`](../tests/test_gate_ledger.py#L642-L654 "anchor:test_existing_ledger_read_error_is_not_clean_unknown"),
    [`test_gate_ledger.py:657-666`](../tests/test_gate_ledger.py#L657-L666 "anchor:test_damaged_newer_row_cannot_resurrect_an_older_pass"),
    [`test_gate_ledger.py:669-682`](../tests/test_gate_ledger.py#L669-L682 "anchor:test_unscoped_damage_does_not_pollute_an_absent_identity_state"),
    [`test_gate_ledger.py:685-694`](../tests/test_gate_ledger.py#L685-L694 "anchor:test_unscoped_damage_does_not_pollute_an_unmatched_identity_state"),
    [`test_gate_ledger.py:697-705`](../tests/test_gate_ledger.py#L697-L705 "anchor:test_later_exact_pass_heals_older_damage"), and
    [`test_gate_ledger.py:708-724`](../tests/test_gate_ledger.py#L708-L724 "anchor:test_non_dict_and_invalid_schema_rows_taint_older_positive_authority").
11. **Every risky normal render carries a dispatch-scoped grant, not merely a
    ledger PASS.** The grant binds the complete primary request and its detailed
    artifact set, effective policy, physical config, resolved topology/attention,
    setup generation/key, worker-policy key, and worker topology. A normal grant
    authorizes exactly one model; a dual-model request is cloned onto complete
    stock residency rather than inheriting primary-only evidence. Driver and
    worker preflight re-derive the bound values, while every normal request
    explicitly selects `required`, `stock`, `operator_off`, `gate_internal`, or
    `capacity_consent`; a missing/stale grant cannot fall through to risky auto
    defaults. `required` means exact current authority was found and attached;
    `stock` means the dispatch policy is non-risky or was forced fully stock;
    `gate_internal` is confined to artifact-bound ceremony renders;
    `capacity_consent` carries a one-dispatch capacity-rescue consent envelope
    with no gate token or ledger authority; and `operator_off` records the
    operator's explicit first-use-Gate bypass rather than asserting PASS.
    Authorization, setup binding, driver validation, and
    worker revalidation are
    [`gate_identity.py:329-477`](../src/dgx_monarch/nodes/gate_identity.py#L329-L477 "anchor:authorize_normal_render"),
    [`render_submit.py:150-369`](../src/dgx_monarch/nodes/render_submit.py#L150-L369 "anchor:_submit_render_guarded"),
    [`mesh_residency.py:87-150`](../src/dgx_monarch/mesh_residency.py#L87-L150 "anchor:assert_normal_render_residency_mode"),
    [`mesh_residency.py:237-369`](../src/dgx_monarch/mesh_residency.py#L237-L369 "anchor:assert_normal_render_residency_grant"),
    [`mesh_rpc.py:225-340`](../src/dgx_monarch/mesh_rpc.py#L225-L340 "anchor:verify_request_artifacts"), and
    [`worker_authorization.py:5-52`](../src/dgx_monarch/actor/worker_authorization.py#L5-L52 "anchor:verify_sample_artifact_authorization").
    Exact precedence/topology, ledger-read refusal, field tampering, and mode
    regressions are
    [`test_gate_orchestration.py:89-115`](../tests/test_gate_orchestration.py#L89-L115 "anchor:test_normal_render_authority_obeys_durable_and_session_precedence"),
    [`test_gate_orchestration.py:118-143`](../tests/test_gate_orchestration.py#L118-L143 "anchor:test_normal_render_pass_is_scoped_to_concrete_topology"),
    [`test_gate_orchestration.py:146-160`](../tests/test_gate_orchestration.py#L146-L160 "anchor:test_normal_render_ledger_read_error_never_uses_cached_pass"),
    [`test_gate_orchestration.py:163-177`](../tests/test_gate_orchestration.py#L163-L177 "anchor:test_normal_render_unscoped_damage_never_uses_cached_pass"),
    [`test_gate_orchestration.py:783-817`](../tests/test_gate_orchestration.py#L783-L817 "anchor:test_dual_model_render_never_reuses_primary_only_pass"),
    [`test_gate_orchestration.py:820-852`](../tests/test_gate_orchestration.py#L820-L852 "anchor:test_pipeline_dual_model_bypasses_primary_pass_and_forces_stock"),
    [`test_gate_orchestration.py:3200-3237`](../tests/test_gate_orchestration.py#L3200-L3237 "anchor:test_dual_model_authorization_never_mints_primary_only_grant"),
    [`test_runtime_remediation.py:121-136`](../tests/test_runtime_remediation.py#L121-L136 "anchor:test_normal_residency_grant_binds_every_reconstructible_boundary"),
    [`test_runtime_remediation.py:139-172`](../tests/test_runtime_remediation.py#L139-L172 "anchor:test_normal_residency_grant_rejects_tampering"),
    [`test_runtime_remediation.py:175-186`](../tests/test_runtime_remediation.py#L175-L186 "anchor:test_risky_normal_render_requires_driver_mode_and_grant"), and
    [`test_runtime_remediation.py:1610-1643`](../tests/test_runtime_remediation.py#L1610-L1643 "anchor:test_worker_rechecks_normal_grant_before_model_load").
12. **Process verdicts cannot resurrect a newly disproved durable PASS.** One
    copy-on-write publication covers normal, Fleet, and sibling tokens. Negative
    verdicts live in a non-evicting revocation map and outrank both the bounded
    positive cache and the ledger for this process. Every token includes the
    first-use-cached, process-lifetime package-source manifest. Driver and
    worker processes therefore cannot exchange authority across same-version
    source identities. Automatic ceremonies publish
    PASS only after token-drift validation; waiters use one bounded single-flight
    claim without recursively acquiring its non-reentrant lock. Publication and
    precedence are
    [`auto_gate.py:64-89`](../src/dgx_monarch/nodes/auto_gate.py#L64-L89 "anchor:record_process_gate_verdicts"),
    [`auto_gate.py:102-103`](../src/dgx_monarch/nodes/auto_gate.py#L102-L103 "anchor:process_gate_verdict_locked"),
    [`auto_gate.py:238-299`](../src/dgx_monarch/nodes/auto_gate.py#L238-L299 "anchor:auto_gate_required"),
    [`auto_gate.py:306-505`](../src/dgx_monarch/nodes/auto_gate.py#L306-L505 "anchor:maybe_auto_gate"), and
    [`fleet_policy.py:25-149`](../src/dgx_monarch/nodes/fleet_policy.py#L25-L149 "anchor:_fleet_worker_policy").
    Damage-triggered cache refusal/reproof, unconsumable-PASS denial, Fleet
    refusal, bounded positive-cache eviction, and non-evicting denial are covered by
    [`test_gate_orchestration.py:180-205`](../tests/test_gate_orchestration.py#L180-L205 "anchor:test_auto_gate_ignores_cached_pass_when_ledger_damage_is_unhealed"),
    [`test_gate_orchestration.py:208-230`](../tests/test_gate_orchestration.py#L208-L230 "anchor:test_auto_gate_does_not_cache_unconsumable_pass_over_retesting"),
    [`test_gate_orchestration.py:3240-3315`](../tests/test_gate_orchestration.py#L3240-L3315 "anchor:test_damaged_real_ledger_cached_pass_requires_exact_reproof"),
    [`test_fleet.py:714-749`](../tests/test_fleet.py#L714-L749 "anchor:test_fleet_cached_pass_cannot_bridge_unhealed_ledger_damage"),
    [`test_gate_orchestration.py:855-878`](../tests/test_gate_orchestration.py#L855-L878 "anchor:test_auto_gate_session_cache_evicts_oldest_context") and
    [`test_gate_orchestration.py:881-895`](../tests/test_gate_orchestration.py#L881-L895 "anchor:test_process_denial_survives_bounded_positive_cache_pressure").
    After the durable RETESTING row and process-local INCONCLUSIVE publication,
    every ceremony establishes its exact setup and requires one setup-token-bound
    `provenance_baseline` from every unique rank before the first unload, load,
    render, or swap proof side effect. The same complete world and setup
    generation must report the driver's first-use-cached package-source digest
    again after the proof and policy-drift check, before any terminal process or
    ledger publication. An injected observer runs only after this mandatory
    check, so it cannot replace the cohort authority. The bracket is
    [`gate_provenance.py:19-106`](../src/dgx_monarch/nodes/gate_provenance.py#L19-L106 "anchor:ProofCohortAttestor") and
    [`gate_ceremony.py:231-356`](../src/dgx_monarch/nodes/gate_ceremony.py#L231-L356 "anchor:gather_ceremony_evidence"), covered by
    [`test_gate_orchestration.py:1494-1551`](../tests/test_gate_orchestration.py#L1494-L1551 "anchor:test_provenance_bracket_precedes_every_gate_pass_publication"),
    [`test_gate_orchestration.py:3418-3474`](../tests/test_gate_orchestration.py#L3418-L3474 "anchor:test_default_provenance_attestor_rejects_incomplete_or_mixed_cohort_preproof"),
    [`test_gate_orchestration.py:3477-3512`](../tests/test_gate_orchestration.py#L3477-L3512 "anchor:test_default_provenance_rpc_failure_never_publishes_a_terminal_row"),
    [`test_gate_orchestration.py:3515-3541`](../tests/test_gate_orchestration.py#L3515-L3541 "anchor:test_post_provenance_source_mismatch_cannot_publish_a_terminal_row"),
    [`test_gate_orchestration.py:3544-3570`](../tests/test_gate_orchestration.py#L3544-L3570 "anchor:test_post_provenance_rejects_setup_generation_drift_before_publication"),
    [`test_gate_orchestration.py:3573-3597`](../tests/test_gate_orchestration.py#L3573-L3597 "anchor:test_injected_provenance_observer_cannot_bypass_mandatory_cohort_check"),
    [`test_gate_orchestration.py:3600-3631`](../tests/test_gate_orchestration.py#L3600-L3631 "anchor:test_fresh_ceremony_establishes_setup_after_denial_before_pre_attestation"), and
    [`test_gate_orchestration.py:3634-3659`](../tests/test_gate_orchestration.py#L3634-L3659 "anchor:test_post_provenance_malformed_row_cannot_publish_a_terminal_row").
    Later normal and Fleet workers independently recompute the same source-bound
    token before model load. Source-token and driver/worker skew are covered by
    [`test_gate_ledger.py:1006-1048`](../tests/test_gate_ledger.py#L1006-L1048 "anchor:test_same_version_source_change_invalidates_after_process_cache_reset"),
    [`test_runtime_remediation.py:1672-1709`](../tests/test_runtime_remediation.py#L1672-L1709 "anchor:test_fleet_grant_rejects_driver_worker_source_manifest_mismatch"), and
    [`test_runtime_remediation.py:1712-1738`](../tests/test_runtime_remediation.py#L1712-L1738 "anchor:test_normal_grant_rejects_driver_worker_source_manifest_mismatch").
13. **v9 audit rows are permanent evidence and can never become authority.**
    Every consented bypass writes a `WAIVER` row naming what was waived, under
    which capability context, from which channel, and a certified rescue load
    writes a `CAPACITY_CERTIFIED` row carrying the certificate summary. Three
    independent checks exclude both from trust. They are
    written under a separate key namespace, so an audit row can never become
    the newest row for a gated combination and silently downgrade a live PASS
    to `unknown`; their `capability_context` carries a `record` discriminator
    no real trust context has, so no contextual lookup can select one; and
    their verdict strings are outside the recognized set, so even a crafted
    context yields no grant. The ledger's recording path constructs and
    validates audit rows. Callers must not write them by hand, because a
    malformed audit row could be treated as damage and invalidate an older
    PASS.
    A waived dispatch receives its stamp only after the process reserves
    space for the immutable facts needed to write its use row. When all 512
    slots are live, the next request
    stays unwaived and the ordinary worker guard refuses if it fires;
    active records are never evicted, and waived output cannot return without
    the records required for its permanent use row. This boundary is
    [`consent_waiver.py:275-318`](../src/dgx_monarch/nodes/consent_waiver.py#L275-L318 "anchor:stamp_request") and
    [`consent_waiver.py:321-343`](../src/dgx_monarch/nodes/consent_waiver.py#L321-L343 "anchor:_remember_audit"), covered by
    [`test_accuracy_waivers.py:675-688`](../tests/test_accuracy_waivers.py#L675-L688 "anchor:test_a_513th_outstanding_waived_dispatch_is_not_stamped").
    The vocabulary, builders and namespace are
    [`gate_audit_vocab.py:20-20`](../src/dgx_monarch/gate_audit_vocab.py#L20-L20 "anchor:AUDIT_KEY_PREFIX") and
    [`gate_audit.py:218-226`](../src/dgx_monarch/gate_audit.py#L218-L226 "anchor:record_waiver"), and the
    neutrality, non-damage, no-grant and no-displacement properties are covered
    by
    [`test_ledger_v9.py:245-267`](../tests/test_ledger_v9.py#L245-L267 "anchor:test_audit_context_can_never_equal_a_trust_context"),
    [`test_ledger_v9.py:270-282`](../tests/test_ledger_v9.py#L270-L282 "anchor:test_audit_rows_are_trust_neutral"),
    [`test_ledger_v9.py:285-294`](../tests/test_ledger_v9.py#L285-L294 "anchor:test_audit_rows_are_not_damage"),
    [`test_ledger_v9.py:297-304`](../tests/test_ledger_v9.py#L297-L304 "anchor:test_audit_context_lookup_never_grants"), and
    [`test_ledger_v9.py:307-324`](../tests/test_ledger_v9.py#L307-L324 "anchor:test_waiver_row_never_becomes_latest_for_a_gated_combo"), and
    [`test_gate_ledger.py:1116-1141`](../tests/test_gate_ledger.py#L1116-L1141 "anchor:test_source_failure_preserves_unscoped_diagnostics_and_audit_rows").
14. **A consent is an availability grant, never an accuracy grant.** A consent
    memo permits a residency change so a render can run. It does not establish
    output accuracy. It cannot clear a quarantine: the authoritative
    persisted decision is checked before residency is resolved, the memo is not
    consulted when that decision is FAIL, and the worker treats an explicit
    stock request as terminal regardless of any consent, so a stale or buggy
    driver push cannot rescue into a quarantined lever. The decision applies
    only within the same canonical capability context. A
    compatible FAIL remains the effective quarantine across later
    `INCONCLUSIVE` or `RETESTING` rows; only a newer explicit `PASS` from the
    current protocol, package, and source may clear it. A scoped FAIL also
    retires when the gate protocol or the package version moves, the same
    binding `gate_ledger` requires before it will select a row at all, so one
    rule governs both readers of these rows; an unscoped FAIL predates that
    binding, has none to compare, and stays. A FAIL stamped onto an
    explicit-slab sibling counts as evidence only when the cross-residency leg
    it names diverged, which is the rule the ceremony applies when it writes
    those rows. A PASS from context A
    cannot clear context B, and a foreign-source terminal row cannot mask the
    FAIL. Without a governing FAIL, `INCONCLUSIVE`, `RETESTING`, unreadable
    state, and damage still deny.
    A pre-gate rescue that reports stock residency owes no certificate only
    when an exact list contains one literal-stock slot report from every unique
    rank in the expected world. Partial, duplicate, malformed, or missing-slot
    evidence unloads the slot and raises the typed uncertified-load refusal; an
    observed subset can never excuse slab certification. The check is
    [`consent_rescue.py:72-111`](../src/dgx_monarch/nodes/consent_rescue.py#L72-L111 "anchor:record_rescue_row") and
    [`consent_rescue.py:114-128`](../src/dgx_monarch/nodes/consent_rescue.py#L114-L128 "anchor:_every_rank_took_stock"), covered by
    [`test_consent_wiring.py:789-804`](../tests/test_consent_wiring.py#L789-L804 "anchor:test_partial_duplicate_or_malformed_stock_rescue_evidence_is_refused").
    Pending/active API rows are projected against one such ledger scan. GET
    hides denied or ambiguous rows without mutating them; accept rechecks and
    returns a conflict. It consumes the pending card and revokes the memo when
    the effective decision is a proven FAIL, including FAIL followed only by
    INCONCLUSIVE or RETESTING. An INCONCLUSIVE or RETESTING decision with no
    governing FAIL preserves that state.
    A later current-source PASS permits a fresh consent, and the older FAIL
    then stays in the ledger only as audit history. These rules are implemented in
    [`consent_quarantine.py:230-303`](../src/dgx_monarch/nodes/consent_quarantine.py#L230-L303 "anchor:quarantine_decisions") and
    [`consent_routes.py:140-180`](../src/dgx_monarch/nodes/consent_routes.py#L140-L180 "anchor:_live_rows"), and covered by
    [`test_consent_endpoints.py:459-477`](../tests/test_consent_endpoints.py#L459-L477 "anchor:test_accept_refuses_a_card_after_persisted_gate_fail"),
    [`test_consent_endpoints.py:480-501`](../tests/test_consent_endpoints.py#L480-L501 "anchor:test_consent_state_omits_quarantined_rows_without_mutating_them"),
    [`test_consent_endpoints.py:504-528`](../tests/test_consent_endpoints.py#L504-L528 "anchor:test_consent_state_uses_one_memo_snapshot_for_rows_and_quarantine"),
    [`test_consent_endpoints.py:531-549`](../tests/test_consent_endpoints.py#L531-L549 "anchor:test_fail_then_pass_allows_the_fresh_consent"),
    [`test_consent_endpoints.py:552-567`](../tests/test_consent_endpoints.py#L552-L567 "anchor:test_pass_in_another_capability_context_does_not_clear_fail"),
    [`test_consent_endpoints.py:570-601`](../tests/test_consent_endpoints.py#L570-L601 "anchor:test_unscoped_pass_cannot_clear_unscoped_fail_for_contextual_consent"),
    [`test_consent_endpoints.py:604-632`](../tests/test_consent_endpoints.py#L604-L632 "anchor:test_stale_contextual_pass_cannot_clear_compatible_historical_fail"),
    [`test_consent_endpoints.py:639-652`](../tests/test_consent_endpoints.py#L639-L652 "anchor:test_retesting_after_fail_preserves_quarantine_and_consumes_card"),
    [`test_consent_endpoints.py:655-676`](../tests/test_consent_endpoints.py#L655-L676 "anchor:test_inconclusive_after_fail_preserves_quarantine_and_revokes"),
    [`test_consent_endpoints.py:679-709`](../tests/test_consent_endpoints.py#L679-L709 "anchor:test_foreign_source_terminal_cannot_mask_older_fail"),
    [`test_consent_wiring.py:399-421`](../tests/test_consent_wiring.py#L399-L421 "anchor:test_unscoped_pass_cannot_clear_unscoped_fail_for_live_projection"),
    [`test_consent_wiring.py:424-449`](../tests/test_consent_wiring.py#L424-L449 "anchor:test_stale_contextual_pass_cannot_clear_fail_for_live_projection"), and
    [`test_consent_wiring.py:453-475`](../tests/test_consent_wiring.py#L453-L475 "anchor:test_foreign_source_terminal_cannot_mask_fail_for_live_projection").
    Consent memo schema 2 persists both current and legacy artifact aliases so
    an older-spelling FAIL cannot be missed. A memo file with any other schema,
    schema 1 included, reads as empty, including its standing auto-rescue bit,
    and is replaced on the next write; the operator must grant each consent again and
    re-enable auto-rescue if desired.
    Memo mutation fsyncs the replacement file and, after atomic rename, the
    containing directory. The rename applies the change, so every later read
    already sees the new memo. A directory-sync failure after it leaves only
    the durability unknown, and the store raises `ConsentStoreError` instead of
    reporting durable success; cancellation keeps its exact identity. A revoke
    therefore never reports durable success while the old grant might return
    after a crash. The publication boundary is
    [`consent_store.py:260-281`](../src/dgx_monarch/consent_store.py#L260-L281 "anchor:_write_atomic") and
    [`consent_store.py:284-298`](../src/dgx_monarch/consent_store.py#L284-L298 "anchor:_fsync_parent_directory"), covered by
    [`test_consent_store.py:122-140`](../tests/test_consent_store.py#L122-L140 "anchor:test_directory_fsync_failure_after_replace_is_loud") and
    [`test_consent_store.py:143-165`](../tests/test_consent_store.py#L143-L165 "anchor:test_directory_fsync_cancellation_is_not_recast_as_success").
    A ceremony FAIL that quarantines a lever also revokes the class C memo that
    covered it; a class K accuracy waiver clears a math guard rather than a
    lever and is not revoked. Both rows remain on disk: the waiver records what
    was allowed; the FAIL
    records what was later proven unsafe and governs trust lookup. Quarantine
    after a certified load fails is covered by
    [`test_ledger_v9.py:509-532`](../tests/test_ledger_v9.py#L509-L532 "anchor:test_a_certified_load_that_later_fails_still_quarantines").
15. **An approximate attention kernel skips the ceremony instead of passing
   it.** The ceremony compares residency under one kernel, so it cannot judge an
   approximate one. Under the Sol-Attn kernel no ceremony runs, no ledger row is
   written, and no lever is quarantined; the automatic path logs one INFO line
   and grants nothing, and the explicit Gate node reports `NOT RUN`, or raises
   under `strict`. Risky residency therefore renders fully stock, and the render
   still needs its class-K accuracy waiver. The skip is
   [`sol_attention.py:177-183`](../src/dgx_monarch/adapters/sol_attention.py#L177-L183 "anchor:sol_ceremony_skip_reason"),
   the automatic path is
   [`auto_gate.py:168-173`](../src/dgx_monarch/nodes/auto_gate.py#L168-L173 "anchor:auto_gate_context"),
   and the node's NOT RUN branch is
   [`gate_node.py:108-119`](../src/dgx_monarch/nodes/gate_node.py#L108-L119 "anchor:gate"),
   covered by `tests/test_accuracy_waivers.py` and `tests/test_sol_attention.py`.

## Grant paths

- A normal-render grant is created only from an exact current PASS and binds
  to the immutable request and setup snapshot. It is rechecked during
  driver artifact preflight, worker request preflight (including a local
  source-token recomputation), and resident-model
  adoption. It proves exactly one primary model; dual-model requests use a
  stock-policy clone. Stock, Gate-internal, capacity-consent, and explicit
  operator-off paths carry their own mode stamps and cannot be mistaken for a
  required grant. The
  dual-model policy, mint, setup-generation attachment, and three validation
  boundaries are
  [`gate_identity.py:277-287`](../src/dgx_monarch/nodes/gate_identity.py#L277-L287 "anchor:model_for_request"),
  [`gate_identity.py:329-477`](../src/dgx_monarch/nodes/gate_identity.py#L329-L477 "anchor:authorize_normal_render"),
  [`render_submit.py:150-369`](../src/dgx_monarch/nodes/render_submit.py#L150-L369 "anchor:_submit_render_guarded"),
  [`mesh_rpc.py:225-340`](../src/dgx_monarch/mesh_rpc.py#L225-L340 "anchor:verify_request_artifacts"),
  [`worker_authorization.py:5-52`](../src/dgx_monarch/actor/worker_authorization.py#L5-L52 "anchor:verify_sample_artifact_authorization"), and
  [`worker_authorization.py:96-106`](../src/dgx_monarch/actor/worker_authorization.py#L96-L106 "anchor:assert_resident_artifact_identity"), with source-skew coverage in
  [`test_runtime_remediation.py:1712-1738`](../tests/test_runtime_remediation.py#L1712-L1738 "anchor:test_normal_grant_rejects_driver_worker_source_manifest_mismatch").

- A persisted runtime grant exists only when contextual lookup selects a
  current row and returns `pass` for a known matching ComfyUI commit and
  package-source manifest. The exact selection checks are
  [`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity"), with
  coverage in
  [`test_gate_ledger.py:215-228`](../tests/test_gate_ledger.py#L215-L228 "anchor:test_lookup_semantics"),
  [`test_gate_ledger.py:247-269`](../tests/test_gate_ledger.py#L247-L269 "anchor:test_pass_is_bound_to_protocol_version_and_capability_context"),
  [`test_gate_ledger.py:346-350`](../tests/test_gate_ledger.py#L346-L350 "anchor:test_recorded_unknown_commit_never_grants_pass"),
  [`test_gate_ledger.py:353-360`](../tests/test_gate_ledger.py#L353-L360 "anchor:test_contextual_pass_is_invalidated_by_package_version"), and
  [`test_gate_ledger.py:1006-1048`](../tests/test_gate_ledger.py#L1006-L1048 "anchor:test_same_version_source_change_invalidates_after_process_cache_reset").
- For automatic first use, `unknown`, `stale`, or uncached `inconclusive` starts
  the canonical ceremony before the full render. A returned PASS is discarded
  if the artifact/context token changed during that ceremony. This is enforced
  at [`auto_gate.py:306-505`](../src/dgx_monarch/nodes/auto_gate.py#L306-L505 "anchor:maybe_auto_gate") and
  covered by
  [`test_gate_orchestration.py:702-726`](../tests/test_gate_orchestration.py#L702-L726 "anchor:test_inconclusive_auto_gate_retries_only_after_context_changes")
  and [`test_gate_orchestration.py:752-780`](../tests/test_gate_orchestration.py#L752-L780 "anchor:test_auto_gate_discards_pass_when_ceremony_snapshot_changed").
- Session reuse is distinct from durable reuse. A successfully completed
  ceremony `PASS` can authorize its exact token in the current driver process
  when the single ledger scan says that bridging is safe (for example, an
  exact PASS whose ComfyUI commit is unknown). It cannot bridge unhealed
  damage, a source mismatch, or an exact `RETESTING` row. If a best-effort final PASS append fails
  after durable revocation, the process publishes an operational denial and
  uses stock residency rather than contradicting that revocation. This is
  enforced at
  [`auto_gate.py:306-505`](../src/dgx_monarch/nodes/auto_gate.py#L306-L505 "anchor:maybe_auto_gate") and, for the best-effort append,
  [`gate_verdict.py:229-254`](../src/dgx_monarch/nodes/gate_verdict.py#L229-L254 "anchor:publish_ceremony_verdict"), covered by
  [`test_gate_ledger.py:490-525`](../tests/test_gate_ledger.py#L490-L525 "anchor:test_retest_guard_atomically_denies_every_context_until_exact_finals"),
  [`test_gate_ledger.py:976-1003`](../tests/test_gate_ledger.py#L976-L1003 "anchor:test_contextual_legacy_authority_without_dgx_source_reads_stale"),
  [`test_gate_orchestration.py:3171-3197`](../tests/test_gate_orchestration.py#L3171-L3197 "anchor:test_known_commit_pass_can_be_session_cached_when_persistence_fails"), and
  [`test_gate_orchestration.py:208-230`](../tests/test_gate_orchestration.py#L208-L230 "anchor:test_auto_gate_does_not_cache_unconsumable_pass_over_retesting").
- An unknown-ComfyUI PASS follows the same session-only rule and never becomes
  a persisted reusable grant. That commit-specific boundary is covered by
  [`test_gate_orchestration.py:729-749`](../tests/test_gate_orchestration.py#L729-L749 "anchor:test_unknown_commit_pass_is_cached_only_for_this_process")
  and
  [`test_gate_ledger.py:346-350`](../tests/test_gate_ledger.py#L346-L350 "anchor:test_recorded_unknown_commit_never_grants_pass").
- Ordinary residency ceremonies attempt to record both normal distributed and
  Fleet world-1 contexts from one effective-policy snapshot. Only the guarded
  auto-to-on equivalence may add sibling rows: a sibling `PASS` needs the full
  `PASS` proof, a sibling `FAIL` needs the cross-residency leg to have
  diverged, and an `INCONCLUSIVE` revokes the siblings, which then read back
  as a retest. A `PASS` or `FAIL` whose slab leg never ran writes no sibling
  row. No-LoRA FSDP instead records only
  its exact normal `fsdp_clean_reload_v1` context; it cannot stamp Fleet or a
  residency sibling that the proof did not exercise. This is enforced at
  [`gate_verdict.py:205-248`](../src/dgx_monarch/nodes/gate_verdict.py#L205-L248 "anchor:publish_ceremony_verdict") and
  [`gate_identity.py:518-577`](../src/dgx_monarch/nodes/gate_identity.py#L518-L577 "anchor:equivalent_slab_mode_contexts"), with coverage
  in [`test_gate_orchestration.py:1673-1704`](../tests/test_gate_orchestration.py#L1673-L1704 "anchor:test_cross_mode_reference_passes_and_restores_slab")
  and
  [`test_gate_orchestration.py:2628-2642`](../tests/test_gate_orchestration.py#L2628-L2642 "anchor:test_pass_under_auto_stamps_the_explicit_on_context"),
  [`test_gate_orchestration.py:2645-2660`](../tests/test_gate_orchestration.py#L2645-L2660 "anchor:test_config_supplied_auto_stamps_only_the_explicit_on_variant"), and
  [`test_gate_orchestration.py:2663-2674`](../tests/test_gate_orchestration.py#L2663-L2674 "anchor:test_pass_with_unvouched_family_never_stamps").

## Deny and quarantine paths

- A `RETESTING` row is a durable multi-context denial, and any newer malformed
  or invalid-schema row blocks older exact positive authority and process-cache
  bridging until a later exact valid row heals it (invariant 10). Fleet and
  normal authorization convert lookup errors, an unreadable ledger's typed read
  error included, to stock residency. The
  denial, positional scan, and fail-closed consumers are
  [`gate_ledger.py:161-190`](../src/dgx_monarch/gate_ledger.py#L161-L190 "anchor:_valid_entry_schema"),
  [`gate_ledger.py:192-234`](../src/dgx_monarch/gate_ledger.py#L192-L234 "anchor:entries_with_integrity"),
  [`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity"),
  [`gate_identity.py:329-477`](../src/dgx_monarch/nodes/gate_identity.py#L329-L477 "anchor:authorize_normal_render"), and
  [`fleet_policy.py:25-149`](../src/dgx_monarch/nodes/fleet_policy.py#L25-L149 "anchor:_fleet_worker_policy").
- Process-local FAIL, INCONCLUSIVE, and ERROR verdicts are retained separately
  from the bounded positive cache. They override an older durable PASS for
  normal and Fleet calls until a later successful exact ceremony atomically
  supersedes the same tokens. Copy-on-write publication and denial-first lookup
  are
  [`auto_gate.py:64-89`](../src/dgx_monarch/nodes/auto_gate.py#L64-L89 "anchor:record_process_gate_verdicts") and
  [`auto_gate.py:102-103`](../src/dgx_monarch/nodes/auto_gate.py#L102-L103 "anchor:process_gate_verdict_locked"), covered by
  [`test_gate_orchestration.py:89-115`](../tests/test_gate_orchestration.py#L89-L115 "anchor:test_normal_render_authority_obeys_durable_and_session_precedence") and
  [`test_gate_orchestration.py:881-895`](../tests/test_gate_orchestration.py#L881-L895 "anchor:test_process_denial_survives_bounded_positive_cache_pressure").

- `unknown`, `stale`, `inconclusive`, ceremony error, and token drift grant
  nothing. Ordinary automatic render and pipeline paths disable both unproved
  residency policies before submission. FSDP has no stock-residency equivalent
  inside the same topology, so its automatic path raises a typed refusal unless
  the exact clean-reload token is PASS. `INCONCLUSIVE` remains cached only to
  avoid a ceremony loop and never enables prefetch. An INCONCLUSIVE whose
  ceremony had no material to compare also grants nothing: its render is
  submitted with both levers off by the same per-dispatch authorization, but it
  takes no lever from other combinations in the process, because it disproved
  nothing about them
  ([`gate_inconclusive.py:106-113`](../src/dgx_monarch/nodes/gate_inconclusive.py#L106-L113 "anchor:record_inconclusive"),
  [docs/TROUBLESHOOTING.md #66](TROUBLESHOOTING.md#66-the-gate-says-inconclusive-on-a-graph-with-no-lora)). Later renders read that classification from
  one of two stores. In this process it is a memo the verdict publication
  refreshes in the same locked step, for whichever ceremony published last, so
  it can never contradict the denial it explains.
  Across processes it is the ledger row, which carries the classification and
  so grants the first-use skip under the whole identity the lookup already
  proves: combination, artifact bytes, comfy commit, capability context, gate
  protocol, release, and package-source manifest. A row without the classification, a retest
  guard, and damage newer than the row each keep the ceremony
  ([`gate_inconclusive.py:259-259`](../src/dgx_monarch/nodes/gate_inconclusive.py#L259-L259 "anchor:row_grants_skip")).
  Only the capability context the ceremony ran under is
  remembered; the Fleet scope and the explicit-on slab siblings revoked in the
  same transaction get no skip
  ([`gate_inconclusive.py:207-207`](../src/dgx_monarch/nodes/gate_inconclusive.py#L207-L207 "anchor:remember_kind")).
  This is enforced at
  [`common.py:288-366`](../src/dgx_monarch/nodes/common.py#L288-L366 "anchor:run_render"),
  [`auto_gate.py:234-299`](../src/dgx_monarch/nodes/auto_gate.py#L234-L299 "anchor:auto_gate_required"),
  [`auto_gate.py:302-501`](../src/dgx_monarch/nodes/auto_gate.py#L302-L501 "anchor:maybe_auto_gate"), and
  [`pipeline.py:282-356`](../src/dgx_monarch/nodes/pipeline.py#L282-L356 "anchor:_push_bound"), with
  coverage in
  [`test_gate_orchestration.py:702-726`](../tests/test_gate_orchestration.py#L702-L726 "anchor:test_inconclusive_auto_gate_retries_only_after_context_changes"),
  [`test_gate_orchestration.py:898-922`](../tests/test_gate_orchestration.py#L898-L922 "anchor:test_full_render_is_submitted_only_after_unproven_paths_are_quarantined"),
  and
  [`test_gate_orchestration.py:1385-1429`](../tests/test_gate_orchestration.py#L1385-L1429 "anchor:test_pipeline_gate_verdict_controls_remaining_depth").
- An aborted ordinary residency ceremony, including artifact drift, turns both
  policy levers off for its own combination before re-raising, with two
  exceptions. A live capacity-rescue consent that covers a render with no LoRA
  stack keeps both on, and a graph that declares FSDP and carries a LoRA stack
  keeps `lora_low_rss` on. An abort that a waivable class K guard raised takes
  no lever. An aborted FSDP proof does not classify the failure as a
  residency quarantine. Early aborts, late post-A
  validation errors, and terminal A/B FAIL each attempt exactly one all-rank
  unload while retaining their exact cleanup authority. Two cases need no
  unload because the ceremony retains no residency: a clean reload
  the driver priced and refused before the ceremony's first durable side
  effect created no residency, and a fleet the driver already evicted releases
  its residency with its processes. Confirmation requires
  exactly one positive `unloaded` response per expected rank; partial or
  malformed responses fail closed. An unconfirmed unload attempts the idempotent
  DIRTY latch up to twice, then marks the handle defunct if the latch cannot publish;
  the path re-raises a typed refusal or preserves the original cancellation.
  Automatic token, scope, or ledger-state rejection of
  a provisional FSDP PASS performs the same cleanup on the exact ceremony
  handle before refusing the full render. The
  abort path, and the price that precedes it, are
  [`gate_ceremony.py:328-450`](../src/dgx_monarch/nodes/gate_ceremony.py#L328-L450 "anchor:gather_ceremony_evidence"),
  [`gate_ceremony.py:191-222`](../src/dgx_monarch/nodes/gate_ceremony.py#L191-L222 "anchor:gather_ceremony_evidence"),
  [`fsdp_reload_price.py:181-215`](../src/dgx_monarch/fsdp_reload_price.py#L181-L215 "anchor:price_fsdp_clean_reload") and
  [`gate_session.py:19-169`](../src/dgx_monarch/nodes/gate_session.py#L19-L169 "anchor:run_identity_ceremony"),
  [`gate_fsdp.py:179-235`](../src/dgx_monarch/nodes/gate_fsdp.py#L179-L235 "anchor:cleanup_aborted_fsdp_proof"),
  [`gate_fsdp.py:241-264`](../src/dgx_monarch/nodes/gate_fsdp.py#L241-L264 "anchor:cleanup_rejected_fsdp_pass"), and
  [`auto_gate.py:306-505`](../src/dgx_monarch/nodes/auto_gate.py#L306-L505 "anchor:maybe_auto_gate"), covered by
  [`test_gate_orchestration.py:2282-2318`](../tests/test_gate_orchestration.py#L2282-L2318 "anchor:test_aborted_fsdp_gate_unloads_every_rank_before_refusing"),
  [`test_gate_orchestration.py:2321-2349`](../tests/test_gate_orchestration.py#L2321-L2349 "anchor:test_aborted_fsdp_gate_latches_dirty_when_all_rank_unload_fails"),
  [`test_gate_orchestration.py:2352-2377`](../tests/test_gate_orchestration.py#L2352-L2377 "anchor:test_fsdp_terminal_fail_latches_dirty_when_final_unload_fails"),
  [`test_gate_orchestration.py:2380-2416`](../tests/test_gate_orchestration.py#L2380-L2416 "anchor:test_fsdp_cleanup_requires_complete_positive_all_rank_evidence"),
  [`test_gate_orchestration.py:2419-2453`](../tests/test_gate_orchestration.py#L2419-L2453 "anchor:test_fsdp_cleanup_retries_dirty_latch_then_retires_if_exhausted"),
  [`test_gate_orchestration.py:2456-2496`](../tests/test_gate_orchestration.py#L2456-L2496 "anchor:test_late_fsdp_validation_baseexception_unloads_once_and_preserves_identity"),
  [`test_gate_orchestration.py:2499-2529`](../tests/test_gate_orchestration.py#L2499-L2529 "anchor:test_late_fsdp_validation_exception_is_typed_after_exact_cleanup"),
  [`test_auto_gate_fsdp_refusal.py:99-161`](../tests/test_auto_gate_fsdp_refusal.py#L99-L161 "anchor:test_auto_rejected_fsdp_pass_cleans_exact_ceremony_handle"), and
  [`test_auto_gate_fsdp_refusal.py:164-192`](../tests/test_auto_gate_fsdp_refusal.py#L164-L192 "anchor:test_auto_rejected_fsdp_pass_respects_completed_shared_cleanup"), plus
  [`test_gate_fsdp_proof.py:206-245`](../tests/test_gate_fsdp_proof.py#L206-L245 "anchor:test_interrupted_cleanup_publication_latches_dirty_without_second_unload") and
  [`test_gate_orchestration.py:2610-2625`](../tests/test_gate_orchestration.py#L2610-L2625 "anchor:test_artifact_drift_in_cross_leg_aborts_and_forces_stock").
- Every fresh model load first publishes one process-rooted
  `FreshLoadOwnership` transaction, before FSDP pin or slab acquisition. Each
  child is handed into that exact owner, and the transaction records the exact
  `ModelStore`, slot attribute, and `StoredModel` candidate before slot
  publication. It disarms only after identity confirms that object in that
  slot. If recovery observes the exact publication, it adopts the resident's
  children instead of closing them beneath the model; otherwise the rooted
  transaction owns failed-load cleanup. The transaction is
  [`store_load_ownership.py:17-99`](../src/dgx_monarch/actor/store_load_ownership.py#L17-L99 "anchor:FreshLoadOwnership"),
  built on exact process-root publication and confirmation in
  [`slab_lifetime.py:84-91`](../src/dgx_monarch/actor/slab_lifetime.py#L84-L91 "anchor:prepublish_resource") and
  [`slab_lifetime.py:112-128`](../src/dgx_monarch/actor/slab_lifetime.py#L112-L128 "anchor:confirm_prepublished_resource").
  Instruction-boundary coverage includes
  [`test_store_ensure.py:30-82`](../tests/test_store_ensure.py#L30-L82 "anchor:test_partial_adoption_record_never_mistakes_an_empty_slot_for_publication"),
  [`test_store_ensure.py:262-315`](../tests/test_store_ensure.py#L262-L315 "anchor:test_fsdp_pin_call_return_interrupt_is_recovered_from_caller_handoff"),
  [`test_store_ensure.py:318-391`](../tests/test_store_ensure.py#L318-L391 "anchor:test_post_load_recovery_interrupt_retains_transaction_until_unload"),
  [`test_store_ensure.py:394-459`](../tests/test_store_ensure.py#L394-L459 "anchor:test_slot_publication_interrupt_adopts_exact_resident_for_safe_reuse"), and
  [`test_store_ensure.py:462-502`](../tests/test_store_ensure.py#L462-L502 "anchor:test_slab_load_return_interrupt_recovers_exact_caller_handoff").
- A failed model load whose cleanup cannot be confirmed leaves process-wide
  cleanup poison. Slab mappings and non-slab resources such as pinned FSDP
  checkpoints are retained at module scope rather than closed while Comfy may
  still refer to them; poison remains even when there is no concrete resource
  to retain. While any ownership or poison is pending, `ModelStore` refuses both
  model load and reuse. Cleanup prepublishes each normal-drop owner, and a
  failed-load path publishes its complete slab/non-slab batch before global
  unload/cache/collection or the first resource close; ownership is forgotten
  only after that exact `close()` returns
  normally. If ownership publication itself is interrupted, process poison is
  already latched and no destructive close begins; the worker must be recycled
  rather than treating the incomplete registry as reusable state.

  Retry authority ends before numeric descriptor release. A failed explicit
  global unload/cache/collection pass, FSDP alias unlink, or slab
  `mmap.close()` `BufferError` leaves the descriptor and mapping under known
  ownership and may be retried after another global unload. Before
  `os.close(fd)`, a checkpoint pin, slab arena, or temporary slab checkpoint
  source owner publishes an uncertain state and invalidates its stored numeric
  FD. Any error or interruption from that point is ambiguous even when a test
  hook knows whether the kernel released the descriptor: the worker never calls
  `close(2)` on that number again, refuses pointer/reuse claims where applicable,
  retains cleanup poison, and requires worker recycle. Retrying could close an
  unrelated resource that reused the same number. `WeightSlab` attempts its
  primary and annex arena closes independently, but one uncertain arena keeps
  the slab retained. Temporary Comfy slab hooks are protected by a separate
  prepublished restore guard. Both process-global attributes must again be the
  exact original objects before the guard releases the slab's close barrier.
  A failed or interrupted restore retains the guard and slab, and a later
  cleanup retries restoration before any arena close.

  A confirmed explicit global unload, cache cleanup, and collection retry may
  close only resources that remain retry-safe and clear their poison. If a
  retry reaches terminal descriptor uncertainty or otherwise remains blocked,
  reset the Attached mesh. Worker status exposes the generic
  `failed_load_cleanup_pending` flag, while
  `retained_failed_load_slabs` counts retained slabs and
  `retained_failed_load_resources` separately counts pins and other non-slab
  owners. FSDP pin construction additionally refuses before opening the
  checkpoint on non-Linux workers or Linux environments without
  `/proc/self/fd`; it never falls back to a racy path reopen. Failed-load
  integration, ownership, refusal, status, platform binding, and release are
  [`comfy_bridge.py:161-267`](../src/dgx_monarch/actor/comfy_bridge.py#L161-L267 "anchor:slab_load"),
  [`comfy_bridge.py:270-329`](../src/dgx_monarch/actor/comfy_bridge.py#L270-L329 "anchor:load_diffusion_model_slab"),
  [`model_store.py:236-336`](../src/dgx_monarch/actor/model_store.py#L236-L336 "anchor:ensure"),
  [`slab_lifetime.py:22-22`](../src/dgx_monarch/actor/slab_lifetime.py#L22-L22 "anchor:_RETAINED_FAILED_LOAD_SLABS"),
  [`slab_lifetime.py:28-28`](../src/dgx_monarch/actor/slab_lifetime.py#L28-L28 "anchor:_RETAINED_FAILED_LOAD_RESOURCES"),
  [`slab_lifetime.py:32-32`](../src/dgx_monarch/actor/slab_lifetime.py#L32-L32 "anchor:_FAILED_LOAD_CLEANUP_POISONED"),
  [`slab_lifetime.py:60-65`](../src/dgx_monarch/actor/slab_lifetime.py#L60-L65 "anchor:cleanup_pending"),
  [`slab_lifetime.py:94-102`](../src/dgx_monarch/actor/slab_lifetime.py#L94-L102 "anchor:retain_failed_load_resource"),
  [`slab_lifetime.py:131-162`](../src/dgx_monarch/actor/slab_lifetime.py#L131-L162 "anchor:_close_or_retain_after_explicit_unload"),
  [`slab_lifetime.py:176-180`](../src/dgx_monarch/actor/slab_lifetime.py#L176-L180 "anchor:close_or_retain_after_explicit_unload"),
  [`slab_lifetime.py:183-185`](../src/dgx_monarch/actor/slab_lifetime.py#L183-L185 "anchor:close_slab_or_retain_after_explicit_unload"),
  [`fsdp_checkpoint_pin.py:26-75`](../src/dgx_monarch/actor/fsdp_checkpoint_pin.py#L26-L75 "anchor:PinnedFsdpCheckpoint"),
  [`fsdp_checkpoint_pin.py:112-123`](../src/dgx_monarch/actor/fsdp_checkpoint_pin.py#L112-L123 "anchor:fsdp_reuse_matches_proof"),
  [`fsdp_checkpoint_pin.py:126-134`](../src/dgx_monarch/actor/fsdp_checkpoint_pin.py#L126-L134 "anchor:_require_linux_proc_fd"),
  [`fsdp_checkpoint_pin.py:137-150`](../src/dgx_monarch/actor/fsdp_checkpoint_pin.py#L137-L150 "anchor:_retain_prepublished_pin"),
  [`fsdp_checkpoint_pin.py:153-189`](../src/dgx_monarch/actor/fsdp_checkpoint_pin.py#L153-L189 "anchor:_discard_partial_fsdp_pin"),
  [`fsdp_checkpoint_pin.py:192-275`](../src/dgx_monarch/actor/fsdp_checkpoint_pin.py#L192-L275 "anchor:pin_fsdp_checkpoint"),
  [`slab_arena.py:20-48`](../src/dgx_monarch/actor/slab_arena.py#L20-L48 "anchor:_OwnedDescriptor"),
  [`slab_arena.py:60-130`](../src/dgx_monarch/actor/slab_arena.py#L60-L130 "anchor:_SlabHookRestore"),
  [`slab_arena.py:156-179`](../src/dgx_monarch/actor/slab_arena.py#L156-L179 "anchor:_finish_prepublished"),
  [`slab_arena.py:182-301`](../src/dgx_monarch/actor/slab_arena.py#L182-L301 "anchor:_Arena"),
  [`slab.py:86-152`](../src/dgx_monarch/actor/slab.py#L86-L152 "anchor:__init__"),
  [`slab.py:161-227`](../src/dgx_monarch/actor/slab.py#L161-L227 "anchor:_read_all"),
  [`slab.py:436-480`](../src/dgx_monarch/actor/slab.py#L436-L480 "anchor:close"),
  [`slab_lifetime.py:203-252`](../src/dgx_monarch/actor/slab_lifetime.py#L203-L252 "anchor:release_after_explicit_unload"),
  [`slab_lifetime.py:255-378`](../src/dgx_monarch/actor/slab_lifetime.py#L255-L378 "anchor:cleanup_failed_load"),
  [`model_store.py:399-442`](../src/dgx_monarch/actor/model_store.py#L399-L442 "anchor:_drop"),
  [`model_store.py:456-456`](../src/dgx_monarch/actor/model_store.py#L456-L456 "anchor:unload_all"),
  [`model_store.py:464-467`](../src/dgx_monarch/actor/model_store.py#L464-L467 "anchor:release_retained_cleanup"), and
  [`model_store.py:475-489`](../src/dgx_monarch/actor/model_store.py#L475-L489 "anchor:snapshot").
  Retention, refusal, and recovery are covered by
  [`test_slab.py:256-279`](../tests/test_slab.py#L256-L279 "anchor:test_close_attempts_annex_after_primary_arena_failure"),
  [`test_slab.py:670-681`](../tests/test_slab.py#L670-L681 "anchor:test_close_retains_live_export_until_retry"),
  [`test_slab.py:684-719`](../tests/test_slab.py#L684-L719 "anchor:test_arena_fd_close_uncertainty_is_terminal_and_never_retried"),
  [`test_slab.py:722-765`](../tests/test_slab.py#L722-L765 "anchor:test_partial_weight_slab_preserves_primary_and_retains_uncertain_arena"),
  [`test_slab.py:920-958`](../tests/test_slab.py#L920-L958 "anchor:test_slab_load_exposed_baseexception_has_durable_owner"),
  [`test_slab_instruction_handoffs.py:524-631`](../tests/test_slab_instruction_handoffs.py#L524-L631 "anchor:test_failed_hook_restore_guards_slab_until_later_confirmed_restore"),
  [`test_store_ensure.py:505-551`](../tests/test_store_ensure.py#L505-L551 "anchor:test_fsdp_pin_close_failure_blocks_reuse_until_exact_resource_retry"),
  [`test_store_ensure.py:554-608`](../tests/test_store_ensure.py#L554-L608 "anchor:test_fsdp_pin_fd_close_is_never_retried_after_release_then_raise"),
  [`test_store_ensure.py:611-656`](../tests/test_store_ensure.py#L611-L656 "anchor:test_fsdp_pin_fd_close_is_never_retried_after_fail_before_release"),
  [`test_store_ensure.py:719-772`](../tests/test_store_ensure.py#L719-L772 "anchor:test_failed_load_cleanup_baseexception_retains_owned_slab"),
  [`test_store_ensure.py:775-844`](../tests/test_store_ensure.py#L775-L844 "anchor:test_non_slab_failed_load_cleanup_poison_requires_confirmed_unload_retry"),
  [`test_store_ensure.py:847-894`](../tests/test_store_ensure.py#L847-L894 "anchor:test_failed_load_resource_stays_open_until_global_cleanup_is_confirmed"),
  [`test_store_ensure.py:897-969`](../tests/test_store_ensure.py#L897-L969 "anchor:test_cleanup_failed_load_prepublishes_and_retains_the_whole_failed_batch"),
  [`test_store_ensure.py:972-1016`](../tests/test_store_ensure.py#L972-L1016 "anchor:test_cleanup_failed_load_publication_interruption_latches_process_poison"),
  [`test_store_ensure.py:1019-1044`](../tests/test_store_ensure.py#L1019-L1044 "anchor:test_retained_only_cleanup_failure_latches_poison"),
  [`test_store_detect_options.py:820-859`](../tests/test_store_detect_options.py#L820-L859 "anchor:test_partial_pin_close_failure_preserves_primary_and_retains_cleanup"), and
  [`test_store_detect_options.py:952-969`](../tests/test_store_detect_options.py#L952-L969 "anchor:test_partial_pin_unlink_failure_retains_fd_until_explicit_cleanup").
  The platform prerequisite is covered by
  [`test_store_detect_options.py:670-675`](../tests/test_store_detect_options.py#L670-L675 "anchor:test_fsdp_pin_requires_linux_proc_fd_before_open_or_alias_creation").

## Version-burn policy

Increment `GATE_PROTOCOL_VERSION` whenever the ceremony or meaning of a trusted
PASS changes. A burn makes older contextual PASS grants stale;
sticky FAIL applies only after current protocol/package/context selection. The
policy and selector are
[`gate_ledger.py:49-49`](../src/dgx_monarch/gate_ledger.py#L49-L49 "anchor:GATE_PROTOCOL_VERSION") and
[`gate_ledger.py:260-343`](../src/dgx_monarch/gate_ledger.py#L260-L343 "anchor:lookup_with_integrity"), and the
v4/v5/v6-to-v7 history, the v7-to-v8, v8-to-v9, and v9-to-v10 burns are covered by
[`test_gate_ledger.py:272-288`](../tests/test_gate_ledger.py#L272-L288 "anchor:test_v5_pass_without_frozen_transaction_inputs_reads_stale") and
[`test_gate_ledger.py:291-308`](../tests/test_gate_ledger.py#L291-L308 "anchor:test_divergent_v6_lineages_read_stale_under_current_protocol"),
[`test_gate_ledger.py:802-830`](../tests/test_gate_ledger.py#L802-L830 "anchor:test_v7_pass_without_fsdp_clean_reload_reads_stale_under_v8"),
[`test_gate_ledger.py:833-860`](../tests/test_gate_ledger.py#L833-L860 "anchor:test_v8_pass_without_the_byte_verify_rule_reads_stale_under_v9"),
[`test_gate_ledger.py:919-943`](../tests/test_gate_ledger.py#L919-L943 "anchor:test_v9_cross_mode_pass_without_repeat_identity_reads_stale_under_v10"), and
[`test_gate_ledger.py:311-343`](../tests/test_gate_ledger.py#L311-L343 "anchor:test_burned_v4_rows_read_stale_not_current_trust").
The v11-to-v12 burn is covered by
[`test_gate_ledger.py:863-887`](../tests/test_gate_ledger.py#L863-L887 "anchor:test_v11_pass_reads_stale_under_v12").

One bump carries every new row shape and PASS-semantic change at once. v9
introduces the `WAIVER` and `CAPACITY_CERTIFIED` audit rows together, with every
field they carry and the two class-K waiver kinds of that release. v10 burns v9
because a v9 cross-mode PASS need not prove its same-residency A/B repeat and v9
authority binds neither the package-source manifest nor the source-matched
proof cohort. Each protocol version has one meaning. A later semantic or
row-format change retires that version and requires a new number. v11 burns
v10 because it adds a class-K kind to the frozen
waiver vocabulary, `waive-known-wrong:sol-attn`, for an approximate attention
kernel that is not identity preserving. A row's audit
fields mean what the vocabulary of its protocol says they mean, so the member
ships with the bump rather than widening v10 after the fact. Every v10 PASS
re-proves once on first use. v12 burns v11 because the v11 slab-cycle proof
accepts a response count without binding each response to its exact
rank/world/current-generation cohort. v13 burns v12 because it adds a fourth
class-K kind to the frozen waiver vocabulary, `waive-known-wrong:shard-quant`,
for the sharded-quant scale bar; invariant 1 cites its burn test,
`test_v12_pass_reads_stale_under_v13`. Rows written under a burned number stay
on disk and stay stale.


## Installation dependency checks

The installation procedure and CI use [the dependency checker](../tools/check_dependencies.py).
Its exception rules have one home in [troubleshooting entry 109](TROUBLESHOOTING.md#109-pip-check-reports-cusparselt-is-not-supported-on-this-platform).
Passing this check grants no model-gate verdict or hardware acceptance.

| Guard | Regression coverage |
|---|---|
| Unexpected pip output or exit status must fail. | [`test_check_dependencies.py:77-93`](../tests/test_check_dependencies.py#L77-L93 "anchor:test_other_results_fail_without_download") |
| Platform, package, download and installed-file mismatches must fail. | [`test_check_dependencies.py:96-118`](../tests/test_check_dependencies.py#L96-L118 "anchor:test_exception_requires_original_wheel") |
| CPU jobs must use the reviewed GitHub-hosted images. | [`test_release_surfaces.py:422-442`](../tests/test_release_surfaces.py#L422-L442 "anchor:test_public_source_has_no_hardware_route") |

Verified-update file ownership follows [troubleshooting entry 110](TROUBLESHOOTING.md#110-verified-update-refuses-color-matcher-test-files).
The color-matcher allowance applies to bundled test files only and grants
no model verdict or runtime-package exception.

| Guard | Regression coverage |
|---|---|
| The color-matcher layout must preserve the complete pinned payload digest without external reads. | [`test_update_color_matcher_layout.py:60-70`](../tests/test_update_color_matcher_layout.py#L60-L70 "anchor:test_complete_layout_keeps_pinned_digest_and_never_reads_external_paths") |
| Other owners, versions, paths, changed files and executable initializers remain rejected. | [`test_update_color_matcher_layout.py:78-118`](../tests/test_update_color_matcher_layout.py#L78-L118 "anchor:test_layout_refuses_every_other_shape") |
| Production package and module shadows remain rejected. | [`test_update_payload_ownership.py:59-63`](../tests/test_update_payload_ownership.py#L59-L63 "anchor:test_foreign_claim_cannot_shadow_or_overlap_pinned_package") |
| Worker attestation must still reject a changed initializer or direct pinned-file claim. | [`test_update_worker_attestation.py:177-182`](../tests/test_update_worker_attestation.py#L177-L182 "anchor:test_worker_attestation_still_rejects_color_matcher_shadow_or_false_claim") |
| Recovery must bind the original checkout, config and target independently of the fixed controller. | [`test_update_recovery_bootstrap.py:90-109`](../tests/test_update_recovery_bootstrap.py#L90-L109 "anchor:test_bound_original_refuses_drift") |
