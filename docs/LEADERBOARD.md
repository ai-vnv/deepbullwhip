# Benchmark Leaderboard

<style>
.lb-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9em;
  margin: 1em 0 2em 0;
}
.lb-table thead {
  background: var(--md-primary-fg-color, #006747);
  color: #fff;
}
.lb-table th {
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
.lb-table th:hover {
  background: var(--md-primary-fg-color--dark, #004d35);
}
.lb-table th .sort-icon::after { content: " \2195"; font-size: 0.7em; opacity: 0.5; }
.lb-table td {
  padding: 8px 14px;
  border-bottom: 1px solid #e0e0e0;
}
.lb-table tbody tr:hover {
  background: var(--md-primary-fg-color--light, #e8f5f0);
}
.lb-table tr.lb-top3 {
  font-weight: 500;
}
.lb-rank { text-align: center; width: 40px; }
.lb-name code {
  background: #f4f4f4;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.88em;
}
.lb-metric { text-align: right; font-variant-numeric: tabular-nums; }
.lb-ref { font-size: 0.85em; }
.lb-ref.lb-none { opacity: 0.3; }
.lb-desc {
  color: #666;
  font-size: 0.9em;
  margin: -0.5em 0 0.5em 0;
}
.lb-protocol {
  background: #f8f9fa;
  border-left: 3px solid var(--md-primary-fg-color, #006747);
  padding: 12px 16px;
  margin: 1em 0 2em 0;
  font-size: 0.88em;
  color: #555;
  line-height: 1.6;
}
.lb-protocol code {
  background: #eef;
  padding: 1px 5px;
  border-radius: 3px;
}
</style>

<div class="lb-protocol">
<strong>Benchmark protocol</strong><br>
Chain: <code>semiconductor_4tier</code> &nbsp;|&nbsp;
Demand: <code>semiconductor_ar1</code> &nbsp;|&nbsp;
T=156 periods &nbsp;|&nbsp;
N=1,000 Monte Carlo paths &nbsp;|&nbsp;
Seed=966<br>
All results reported at the most upstream echelon (E4).
Sorted by Total Cost (lower is better). Click any column header to re-sort.
</div>

<h3>Forecaster Leaderboard</h3>
<p class="lb-desc">Fixed policy: <code>order_up_to</code> &nbsp;|&nbsp; Fixed demand: <code>semiconductor_ar1</code></p>

<table class="lb-table sortable">
  <thead><tr><th class="lb-rank">#</th><th>Component</th><th>Contributor</th><th>Reference</th><th class="lb-metric">Cum. BWR <span class="sort-icon"></span></th><th class="lb-metric">Fill Rate <span class="sort-icon"></span></th><th class="lb-metric">Total Cost <span class="sort-icon"></span></th></tr></thead>
  <tbody>
    <tr class="lb-top3"><td class="lb-rank">&#129351;</td><td class="lb-name"><code>deepar</code></td><td>M. Arief</td><td><span class="lb-ref"><a href="https://doi.org/10.1016/j.ijforecast.2019.07.001">Salinas et al. 2020</a></span></td><td class="lb-metric">428.4</td><td class="lb-metric">80.1%</td><td class="lb-metric">1,739.9</td></tr>
    <tr class="lb-top3"><td class="lb-rank">&#129352;</td><td class="lb-name"><code>naive</code></td><td>M. Arief</td><td><span class="lb-ref lb-none">&mdash;</span></td><td class="lb-metric">428.4</td><td class="lb-metric">80.1%</td><td class="lb-metric">1,739.9</td></tr>
    <tr class="lb-top3"><td class="lb-rank">&#129353;</td><td class="lb-name"><code>exponential_smoothing</code></td><td>M. Arief</td><td><span class="lb-ref lb-none">&mdash;</span></td><td class="lb-metric">444.1</td><td class="lb-metric">82.6%</td><td class="lb-metric">1,907.4</td></tr>
    <tr><td class="lb-rank">4</td><td class="lb-name"><code>moving_average</code></td><td>M. Arief</td><td><span class="lb-ref lb-none">&mdash;</span></td><td class="lb-metric">448.1</td><td class="lb-metric">82.8%</td><td class="lb-metric">1,957.6</td></tr>
  </tbody>
</table>

<h3>Policy Leaderboard</h3>
<p class="lb-desc">Fixed forecaster: <code>naive</code> &nbsp;|&nbsp; Fixed demand: <code>semiconductor_ar1</code></p>

<table class="lb-table sortable">
  <thead><tr><th class="lb-rank">#</th><th>Component</th><th>Contributor</th><th>Reference</th><th class="lb-metric">Cum. BWR <span class="sort-icon"></span></th><th class="lb-metric">Fill Rate <span class="sort-icon"></span></th><th class="lb-metric">NS Amp. <span class="sort-icon"></span></th><th class="lb-metric">Total Cost <span class="sort-icon"></span></th></tr></thead>
  <tbody>
    <tr class="lb-top3"><td class="lb-rank">&#129351;</td><td class="lb-name"><code>proportional_out</code></td><td>M. Arief</td><td><span class="lb-ref"><a href="https://doi.org/10.1080/0020754031000114743">Disney &amp; Towill 2003</a></span></td><td class="lb-metric">47.4</td><td class="lb-metric">64.2%</td><td class="lb-metric">669.4</td><td class="lb-metric">1,123.5</td></tr>
    <tr class="lb-top3"><td class="lb-rank">&#129352;</td><td class="lb-name"><code>order_up_to</code></td><td>M. Arief</td><td><span class="lb-ref"><a href="https://doi.org/10.1287/mnsc.46.3.436.12069">Chen et al. 2000</a></span></td><td class="lb-metric">428.4</td><td class="lb-metric">80.1%</td><td class="lb-metric">3,199.6</td><td class="lb-metric">1,739.9</td></tr>
    <tr class="lb-top3"><td class="lb-rank">&#129353;</td><td class="lb-name"><code>constant_order</code></td><td>M. Arief</td><td><span class="lb-ref lb-none">&mdash;</span></td><td class="lb-metric">0.0</td><td class="lb-metric">2.6%</td><td class="lb-metric">17.4</td><td class="lb-metric">2,261.5</td></tr>
    <tr><td class="lb-rank">4</td><td class="lb-name"><code>smoothing_out</code></td><td>M. Arief</td><td><span class="lb-ref lb-none">&mdash;</span></td><td class="lb-metric">258.9</td><td class="lb-metric">88.3%</td><td class="lb-metric">13,120.0</td><td class="lb-metric">4,035.5</td></tr>
  </tbody>
</table>

<h3>Demand Generator Leaderboard</h3>
<p class="lb-desc">Fixed policy: <code>order_up_to</code> &nbsp;|&nbsp; Fixed forecaster: <code>naive</code></p>

<table class="lb-table sortable">
  <thead><tr><th class="lb-rank">#</th><th>Component</th><th>Contributor</th><th>Reference</th><th class="lb-metric">Cum. BWR <span class="sort-icon"></span></th><th class="lb-metric">Total Cost <span class="sort-icon"></span></th></tr></thead>
  <tbody>
    <tr class="lb-top3"><td class="lb-rank">&#129351;</td><td class="lb-name"><code>beer_game</code></td><td>M. Arief</td><td><span class="lb-ref"><a href="https://doi.org/10.1287/mnsc.35.3.321">Sterman 1989</a></span></td><td class="lb-metric">227.3</td><td class="lb-metric">393.2</td></tr>
    <tr class="lb-top3"><td class="lb-rank">&#129352;</td><td class="lb-name"><code>arma</code></td><td>M. Arief</td><td><span class="lb-ref lb-none">&mdash;</span></td><td class="lb-metric">2,228.2</td><td class="lb-metric">1,738.7</td></tr>
    <tr class="lb-top3"><td class="lb-rank">&#129353;</td><td class="lb-name"><code>semiconductor_ar1</code></td><td>M. Arief</td><td><span class="lb-ref"><a href="https://www.wsts.org">WSTS</a></span></td><td class="lb-metric">428.4</td><td class="lb-metric">1,739.9</td></tr>
  </tbody>
</table>

---

<p style="font-size: 0.8em; color: #999; margin-top: 2em;">
Auto-generated by <code>benchmarks/run_leaderboard.py</code>.
To reproduce: <code>python benchmarks/run_leaderboard.py</code>
</p>

<script>
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("table.sortable th").forEach(function (th) {
    th.addEventListener("click", function () {
      var table = th.closest("table");
      var tbody = table.querySelector("tbody");
      var rows = Array.from(tbody.querySelectorAll("tr"));
      var idx = Array.from(th.parentNode.children).indexOf(th);
      var asc = th.dataset.sortDir !== "asc";
      th.dataset.sortDir = asc ? "asc" : "desc";
      // Reset other headers
      th.parentNode.querySelectorAll("th").forEach(function(h) {
        if (h !== th) delete h.dataset.sortDir;
      });
      rows.sort(function (a, b) {
        var av = a.children[idx].textContent.replace(/,/g, "").replace(/%/g, "");
        var bv = b.children[idx].textContent.replace(/,/g, "").replace(/%/g, "");
        var an = parseFloat(av), bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      });
      rows.forEach(function (r) { tbody.appendChild(r); });
    });
  });
});
</script>
