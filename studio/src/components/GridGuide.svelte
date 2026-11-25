<script lang="ts">
  // The Grids how-to and its worked example: static text, no props.
</script>

<div class="surface grid-guide">
  <div>
    <h2>What a grid does</h2>
    <p>
      A grid runs the workflow in Create several times and changes only the inputs you pick, so you
      can see what each one does to the result. Each run is a cell, and each cell is its own job in
      Queue.
    </p>
    <p>
      For example, steps [20, 30, 40] on Axis A and cfg [3.5, 5] on Axis B make 3 × 2 = 6 cells with
      the same prompt and seed. Cell 1 is 20 steps at CFG 3.5, cell 2 is 20 steps at CFG 5, cell 3
      is 30 steps at CFG 3.5, and so on to cell 6, 40 steps at CFG 5. The result is six images, any
      two of which open side by side: if 30 and 40 steps look alike, 30 steps is enough for this
      prompt.
    </p>
    <h2>How to run a grid</h2>
    <ol>
      <li>
        Load a workflow in Create and set its inputs there. Each cell is that Create draft with the
        axis values filled in.
      </li>
      <li>
        Pick an input for Axis A and list its values as a JSON array, for example steps 20, 30, 40
        as [20, 30, 40].
      </li>
      <li>
        Axis B is optional and multiplies the cells: three values on Axis A and two on Axis B make
        six cells.
      </li>
      <li>Press Validate every cell, tick the review box, then press Submit reviewed grid.</li>
      <li>
        When a cell finishes, press Inspect full submission to list its outputs, then View one or
        Compare it with the output in view.
      </li>
    </ol>
  </div>
  <div>
    <h2>Your first comparison: 20 steps against 40</h2>
    <ol class="first-comparison">
      <li>
        In Create, load a text to image workflow and write a prompt. Leave the other inputs as they
        are.
      </li>
      <li>
        Here, choose steps for Axis A and write [20, 40]. Leave Axis B on No second axis and Seed
        policy on Fixed draft seeds, so both cells use the seed shown in Create.
      </li>
      <li>
        Press Validate every cell. The preview reads 2 cells · 2 jobs. Tick the review box and press
        Submit reviewed grid.
      </li>
      <li>
        When both cells finish, press Inspect full submission on cell 1 and View its output. Then
        press Inspect full submission on cell 2 and Compare on its output.
      </li>
      <li>
        The viewer below the cells shows the two images side by side, and Parameter differences
        lists what the two jobs recorded differently. Only the steps changed, so any difference in
        the pictures comes from the steps.
      </li>
    </ol>
  </div>
  <p class="muted">
    Each cell passes normal backend admission. Comfy's queue controls execution; Studio does not run
    a second scheduler. More cells increase total compute and memory residency time.
  </p>
</div>

<style>
  /* Two columns on a wide page, so the form stays near the top; one column on a narrow one. */
  .grid-guide {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
    gap: 0 32px;
  }
  .grid-guide > p {
    grid-column: 1 / -1;
    margin-bottom: 0;
  }
</style>
