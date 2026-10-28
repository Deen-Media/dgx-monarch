<script lang="ts">
  // The Grids how-to: static text, no props.
</script>

<div class="surface grid-guide">
  <h2>How to run a grid</h2>
  <ol>
    <li>
      Load a workflow in Create and set its inputs there. Each cell is that Create draft with the
      axis values filled in.
    </li>
    <li>
      Pick an input for Axis A and list its values as a JSON array, for example steps 20, 30, 40 as
      [20, 30, 40].
    </li>
    <li>
      Axis B is optional and multiplies the cells: three values on Axis A and two on Axis B make six
      cells.
    </li>
    <li>Press Validate every cell, tick the review box, then press Submit reviewed grid.</li>
    <li>
      When a cell finishes, press Inspect full submission to list its outputs, then View one or
      Compare it with the output in view.
    </li>
  </ol>
  <p class="muted">
    Each cell passes normal backend admission. Comfy's queue controls execution; Studio does not run
    a second scheduler. More cells increase total compute and memory residency time.
  </p>
</div>
