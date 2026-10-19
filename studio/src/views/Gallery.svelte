<script lang="ts">
  import type { Artifact, Job, ReuseArtifact } from '../contracts';
  import { formatTime } from '../model';
  import { exportJSON } from '../files';
  let {
    allArtifacts,
    jobs,
    projectId,
    selected,
    navigate,
    onselect,
    oncompare,
    reuse,
    setNotice,
  }: {
    allArtifacts: Artifact[];
    jobs: Job[];
    projectId: string;
    selected: Artifact | null;
    navigate: (view: string) => void;
    onselect: (artifact: Artifact) => void;
    oncompare: (artifact: Artifact) => void;
    reuse: ReuseArtifact;
    setNotice: (message: string) => void;
  } = $props();
  let galleryQuery = $state(''),
    galleryKind = $state('all'),
    gallerySort = $state('newest'),
    galleryPage = $state(0);
  let artifacts = $derived(
    allArtifacts
      .filter(
        (a) =>
          (galleryKind === 'all' || a.kind === galleryKind) &&
          (!projectId ||
            a.project_id === projectId ||
            jobs.find((j) => j.id === a.job_id)?.submission.project_id === projectId) &&
          JSON.stringify(a).toLowerCase().includes(galleryQuery.toLowerCase()),
      )
      .sort((a, b) =>
        gallerySort === 'newest' ? b.created_at - a.created_at : a.created_at - b.created_at,
      ),
  );
</script>

<section class="page-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">YOUR OUTPUT LIBRARY</span>
      <h1>Gallery <span class="muted">{artifacts.length}</span></h1>
    </div>
    <button onclick={() => navigate('workspace')}>Back to create ↗</button>
  </div>
  <div class="filter-bar">
    <input
      type="search"
      bind:value={galleryQuery}
      oninput={() => (galleryPage = 0)}
      placeholder="Search filename, model, date or metadata…"
      aria-label="Search gallery"
    /><select
      bind:value={galleryKind}
      onchange={() => (galleryPage = 0)}
      aria-label="Filter media kind"
      ><option value="all">All media</option><option value="image">Images</option><option
        value="video">Video</option
      ></select
    ><select bind:value={gallerySort} aria-label="Sort gallery"
      ><option value="newest">Newest first</option><option value="oldest">Oldest first</option
      ></select
    >
  </div>
  <div class="gallery-grid">
    {#each artifacts.slice(galleryPage * 40, (galleryPage + 1) * 40) as artifact}<article
        class="artifact-card"
      >
        <button
          class="artifact-preview"
          onclick={() => {
            onselect(artifact);
            navigate('workspace');
          }}
          aria-label={`View ${artifact.filename}`}
          ><span>{artifact.kind === 'video' ? '▷' : '▧'}</span><small
            >{artifact.kind ?? 'output'} · Load selected media</small
          ></button
        >
        <h3>{artifact.filename ?? artifact.id}</h3>
        <p>{formatTime(artifact.created_at)}</p>
        <div class="workflow-actions">
          <button onclick={() => void reuse(artifact)}>Reuse</button><button
            onclick={() => {
              if (!selected) {
                onselect(artifact);
                setNotice('First comparison output selected. Choose a second.');
              } else {
                oncompare(artifact);
                navigate('workspace');
              }
            }}>Compare</button
          ><button onclick={() => exportJSON(artifact, `${artifact.id}.json`)}>Metadata</button>
        </div>
      </article>{/each}
  </div>
  {#if !artifacts.length}<div class="empty-state">
      <h2>Your output library starts here.</h2>
      <p>Completed images and videos stay linked to their recorded settings.</p>
    </div>{/if}
  <div class="pagination">
    <button disabled={galleryPage === 0} onclick={() => galleryPage--}>Previous</button><span
      >Page {galleryPage + 1} · 40 items per page</span
    ><button disabled={(galleryPage + 1) * 40 >= artifacts.length} onclick={() => galleryPage++}
      >Next</button
    >
  </div>
</section>
