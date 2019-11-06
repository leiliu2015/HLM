---


---

<h1 id="hlm">HLM</h1>
<p>Heterogeneous Loop Model, shortened as HLM, is a physically sound method to generate three-dimensional (3D) chromosome structures from Hi-C data. It is effectively a multiblock copolymer model in which monomer-monomer interactions (loops) are harmonically restrained with varying interaction strength (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>k</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow><annotation encoding="application/x-tex">k_{ij}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.980548em; vertical-align: -0.286108em;"></span><span class="mord"><span style="margin-right: 0.03148em;" class="mord mathdefault">k</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03148em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span></span></span></span></span>). Details about this method can be found in our <a href="https://www.cell.com/biophysj/fulltext/S0006-3495(19)30540-5">paper</a> (DOI: 10.1016/j.bpj.2019.06.032). The whole modeling includes two steps.</p>
<ol>
<li>Built a interaction strength matrix <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>k</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{k_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span style="margin-right: 0.03148em;" class="mord mathdefault">k</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03148em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> based on a contact frequency matrix <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>p</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{p_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span class="mord mathdefault">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span>.</li>
<li>Run Molecular Dynamics (MD) simulations with the parameters <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>k</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{k_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span style="margin-right: 0.03148em;" class="mord mathdefault">k</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03148em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> to generate an ensemble of 3D structures.</li>
</ol>
<h3 id="system-requirements">System Requirements</h3>
<p>The code was tested on ubuntu 14.04/16.04 LTS. We recommand <a href="https://www.anaconda.com/distribution/">Anaconda</a> to manage the Python environment (Python 2.7) and other required packages (NumPy, SciPy, Numba, and scikit-learn). We used <a href="gnuplot.sourceforge.net">Gnuplot</a> to visualize the results, and <a href="http://espressomd.org/wordpress/">ESPResSo 3.3.1 package</a> to perform MD simulations (<em>optional</em>). Please refer to their websites for the instructions of installation.</p>
<h3 id="file-description">File Description</h3>
<ul>
<li>toyModel/
<ul>
<li><a href="toyModel/L2-gnm.kij.reference">L2-gnm.kij.reference</a> (The <em>true</em> values of <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>k</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{k_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span style="margin-right: 0.03148em;" class="mord mathdefault">k</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03148em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span>)</li>
<li><a href="toyModel/L2-gnm.t0-0.rc1.0.cm">L2-gnm.t0-0.rc1.0.cm</a> (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>p</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{p_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span class="mord mathdefault">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> based on MD simulations, which is used as the input of HLM)</li>
<li><a href="toyModel/hlm.py">hlm.py</a> (A Python script to infer <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>k</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{k_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span style="margin-right: 0.03148em;" class="mord mathdefault">k</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03148em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> from <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>p</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{p_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span class="mord mathdefault">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span>)</li>
<li><a href="toyModel/hlm.gnu">hlm.gnu</a> (A Gnuplot script to plot the results)</li>
</ul>
</li>
<li>GM12878-chr5-90-100Mb-50kb/
<ul>
<li><a href="GM12878-chr5-90-100Mb-50kb/chr5_50kb.090-100Mb.Hi-C.cm">chr5_50kb.090-100Mb.Hi-C.cm</a> (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>p</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{p_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span class="mord mathdefault">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> measured by <a href="https://www.cell.com/cell/fulltext/S0092-8674(14)01497-4?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867414014974%3Fshowall%3Dtrue">Rao <em>et al.</em></a> with Hi-C, which is used as the input of HLM)</li>
<li><a href="GM12878-chr5-90-100Mb-50kb/hlm.py">hlm.py</a> (A Python script to infer <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>k</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{k_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span style="margin-right: 0.03148em;" class="mord mathdefault">k</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03148em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> from <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>p</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{p_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span class="mord mathdefault">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span>)</li>
<li><a href="GM12878-chr5-90-100Mb-50kb/hlm.gnu">hlm.gnu</a> (A Gnuplot script to plot the results)</li>
<li>MD/
<ul>
<li><a href="GM12878-chr5-90-100Mb-50kb/MD/hlm_espresso.tcl">hlm_espresso.tcl</a> (A TCL script used as the input of ESPResSo)</li>
<li><a href="GM12878-chr5-90-100Mb-50kb/MD/chr5_50kb.090-100Mb.iniCfg">chr5_50kb.090-100Mb.iniCfg</a> (A initial input configuration read by the TCL script)</li>
<li><a href="GM12878-chr5-90-100Mb-50kb/MD/chr5_50kb.090-100Mb.HLM-MD.cm">chr5_50kb.090-100Mb.HLM-MD.cm</a> (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>p</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{p_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span class="mord mathdefault">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> based on 3D structures generated by MD simulations)</li>
</ul>
</li>
</ul>
</li>
<li><a href="clearAll.sh">clearAll.sh</a> (A Bash script to remove all the outputs)</li>
</ul>
<h3 id="user-guide">User Guide</h3>
<p>We provides two examples to illustrate how HLM works. The first one in the <a href="toyModel/">toyModel</a> folder uses a chain with two nested loops to demonstrate how to infer <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>k</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{k_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span style="margin-right: 0.03148em;" class="mord mathdefault">k</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03148em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> from <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>p</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{p_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span class="mord mathdefault">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> (see Fig. S4 in the <a href="https://www.cell.com/biophysj/fulltext/S0006-3495(19)30540-5">paper</a> for details). To run this demo, type the following lines at the root of the repository in your terminal.</p>
<pre><code>$ cd ./toyModel
$ python hlm.py L2-gnm.t0-0.rc1.0.cm
$ gnuplot -persist hlm.gnu
</code></pre>
<p>The second example in the <a href="GM12878-chr5-90-100Mb-50kb/">GM12878-chr5-90-100Mb-50kb</a> folder shows the pipeline to build 3D structures of a 10-Mb region on chromosome 5 at a resolution of 50 kb (see Fig. 1 in our <a href="https://www.cell.com/biophysj/fulltext/S0006-3495(19)30540-5">paper</a>). The first step of this demo is very similar to the previous one.</p>
<pre><code>$ cd ./GM12878-chr5-90-100Mb-50kb
$ python hlm.py chr5_50kb.090-100Mb.Hi-C.cm
$ gnuplot -persist hlm.gnu
</code></pre>
<p>The output values of <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>k</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{k_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span style="margin-right: 0.03148em;" class="mord mathdefault">k</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03148em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> will be stored in a file called <code>chr5_50kb.090-100Mb.oeL400W2.km</code>. As the second step (<em>optional</em>), we run MD simulation with those parameters using ESPResSo.</p>
<pre><code>$ cd ./MD
$ cp ../chr5_50kb.090-100Mb.oeL400W2.km ./
$ Espresso hlm_espresso.tcl
</code></pre>
<p>It takes a few hours to finish the simulation with a single CPU, which generates <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mn>1</mn><msup><mn>0</mn><mn>4</mn></msup></mrow><annotation encoding="application/x-tex">10^{4}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.814108em; vertical-align: 0em;"></span><span class="mord">1</span><span class="mord"><span class="mord">0</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.814108em;"><span class="" style="top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">4</span></span></span></span></span></span></span></span></span></span></span></span></span> structures of the chromatin chain. Next, one can compute the contact frequency matrix <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">{</mo><msub><mi>p</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\{p_{ij}\}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mopen">{</span><span class="mord"><span class="mord mathdefault">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathdefault mtight">i</span><span style="margin-right: 0.05724em;" class="mord mathdefault mtight">j</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">}</span></span></span></span></span> based on these structures. To save your time, we have provided the results in the <a href="GM12878-chr5-90-100Mb-50kb/MD">MD</a> folder. All the output files will be deleted by typing <code>$ bash ./clearAll.sh</code> at the root of the repository. If you are interested in HLM, or have further questions about it, please send an email to Lei Liu (<a href="mailto:leiliu2015@163.com">leiliu2015@163.com</a>).</p>

