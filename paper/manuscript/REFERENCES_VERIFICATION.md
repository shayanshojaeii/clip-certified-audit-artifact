# References verification log for `paper/satml2027/references.bib` and `references_extra.bib`

Section 5 (added 2026-09-27) covers the five entries of `references_extra.bib`, which `main.tex` also loads; `tests/test_satml2027_manuscript_sources.py` fails if any entry of either file lacks a record here.

- Date: 2026-09-25
- Scope: every key in `paper/satml2027/references.bib` (36 entries: 17 carried
  over from `paper/iclr2027-template/iclr2027/references.bib` with keys
  unchanged, `nirala2024ovc` replaced by its peer-reviewed SaTML 2024 record,
  and 19 new keys).
- Method: each entry was checked against an official page (arXiv abstract
  page, PMLR, NeurIPS proceedings and its official BibTeX endpoint, CVF Open
  Access, OpenReview API, OUP) or the publisher-deposited Crossref DOI record
  when the publisher landing page refused automated access. No field was
  written unless it appeared in one of the sources listed for that key; fields
  that could not be confirmed were omitted and the entry marked PARTIAL.
- Status vocabulary:
  - CONFIRMED: every field present in the entry matches the cited source(s).
  - PARTIAL: all fields present match a source, but a customary field
    (named in the notes) was omitted because no official source confirmed it,
    or one field (named) could not be re-checked.
  - UNVERIFIED: no official source reached. (None.)
- Access notes (relevant to reproducing this check):
  - DBLP (`dblp.org/search/publ/api`) is behind an Anubis bot challenge and
    returned "Access Denied" / "Making sure you're not a bot" to every request,
    including curl with a browser user agent. DBLP was therefore not usable;
    Crossref DOI records and OpenAlex were used instead where needed.
  - `openaccess.thecvf.com` returns HTTP 403 to the WebFetch tool but HTTP 200
    to curl with a browser user agent; the CVF pages below were fetched with
    curl and parsed locally (`papertitle`, `authors`, `bibref` blocks).
  - OpenReview's web UI and its direct `notes?id=` API endpoints redirect to a
    browser challenge; the `notes/search` API endpoint returned full records
    (used for `carlini2023dds` and `hu2022lora`).
  - IEEE Xplore, Springer Link, PNAS, SAGE, and Annual Reviews landing pages
    returned 403/empty/IdP redirects; the corresponding Crossref DOI records
    (deposited by those publishers) were used.
  - Semantic Scholar API returned HTTP 429 (rate limit) and was not used.

## 1. Per-entry verification table

| Key | URL(s) checked | Fields confirmed | Status |
|---|---|---|---|
| `xia2026feature` | https://arxiv.org/abs/2601.16200 | title; authors (Song Xia, Meiwen Ding, Chenqi Kong, Wenhan Yang, Xudong Jiang); arXiv id; version history v1 22 Jan 2026, v2 27 Jan 2026, v3 19 May 2026 (matches `note`); Comments = "Under review" (no venue) | CONFIRMED |
| `maurer2009empirical` | https://www.cs.mcgill.ca/~colt2009/papers/012.pdf (PDF fetched; page 1 text extracted with pdftotext) | title; authors (Andreas Maurer; Massimiliano Pontil); URL live. Venue string is not printed on the page; it follows from the official COLT 2009 proceedings host (COLT 2009 = 22nd Annual Conference on Learning Theory) | CONFIRMED (venue via official COLT 2009 site host) |
| `chowers2026bug` | https://openaccess.thecvf.com/content/CVPR2026/html/Chowers_Is_the_Modality_Gap_a_Bug_or_a_Feature_A_CVPR_2026_paper.html (curl, HTTP 200); https://arxiv.org/abs/2603.29080; https://cvpr.thecvf.com/virtual/2026/poster/37667 | CVF bibref: author = {Chowers, Rhea and Naparstek, Oshri and Barzelay, Udi and Weiss, Yair}; booktitle = Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); month = June; year = 2026; pages = 30288-30298. PDF link `/content/CVPR2026/papers/Chowers_..._CVPR_2026_paper.pdf` present on the page. arXiv v2 dated 27 Apr 2026. Change vs. ICLR bib: added `month = {June}` | CONFIRMED |
| `liang2022mind` | https://proceedings.neurips.cc/paper_files/paper/2022/hash/702f4db7543a7432431df588d57bc7c9-Abstract-Conference.html | title; authors (Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, James Y Zou); "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)" Main Conference Track | CONFIRMED |
| `schrodi2025twoeffects` | https://arxiv.org/abs/2404.07983 (Comments: "ICLR 2025 (Oral)"); OpenReview id uAFHCZRmXk could not be fetched (api.openreview.net search returned no match; api2.openreview.net and openreview.net redirect to a browser challenge) | title; authors (Simon Schrodi, David T. Hoffmann, Max Argus, Volker Fischer, Thomas Brox); venue ICLR 2025 | PARTIAL (OpenReview forum id in `url` not re-checked) |
| `fahim2024contrastive` | https://arxiv.org/abs/2405.18570 | title; authors (Abrar Fahim, Alex Murphy, Alona Fyshe); arXiv id; latest version v3, 6 Jun 2024; no journal-ref | CONFIRMED |
| `li2025closing` | https://arxiv.org/abs/2507.19054 | title; authors (Binxu Li, Yuhui Zhang, Xiaohan Wang, Weixin Liang, Ludwig Schmidt, Serena Yeung-Levy); arXiv id; v1 25 Jul 2025; no journal-ref | CONFIRMED |
| `dong2025stabilizing` | https://api.crossref.org/works/10.1145/3690624.3709296 (ACM-deposited record) | title; authors (Junhao Dong, Piotr Koniusz, Xinghua Qu, Yew-Soon Ong); container "Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1"; pages 236-247; 2025; DOI; publisher ACM | CONFIRMED |
| `zhu2025cola` | https://proceedings.neurips.cc/paper_files/paper/2025/hash/197772120f07c657d0ed9dd7538fc1a4-Abstract-Conference.html | title; authors (Xingyu Zhu, Beier Zhu, Shuo Wang, Kesen Zhao, Hanwang Zhang); "Advances in Neural Information Processing Systems 38 ... (NeurIPS 2025) Main Conference Track" | CONFIRMED |
| `dong2025pathsimplices` | https://proceedings.mlr.press/v267/dong25f.html | PMLR BibTeX: title; authors; booktitle Proceedings of the 42nd International Conference on Machine Learning; pages 14061--14078; volume 267; series PMLR; year 2025; publisher PMLR | CONFIRMED |
| `dong2025subspaces` | https://openaccess.thecvf.com/content/ICCV2025/html/Dong_Robustifying_Zero-Shot_Vision_Language_Models_by_Subspaces_Alignment_ICCV_2025_paper.html (curl, HTTP 200) | CVF bibref: authors (Dong, Koniusz, Feng, Zhang, Zhu, Liu, Qu, Ong); booktitle Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV); October 2025; pages 21037-21047 | CONFIRMED |
| `schlarmann2024robustclip` | https://proceedings.mlr.press/v235/schlarmann24a.html | PMLR BibTeX: title; authors; 41st ICML; pages 43685--43704; volume 235; PMLR; 2024 | CONFIRMED |
| `nirala2024ovc` (replaced) | https://api.crossref.org/works/10.1109/SaTML59370.2024.00019 (IEEE-deposited record); https://api.openalex.org/works?search=Fast%20Certification%20of%20Vision-Language%20Models%20Using%20Incremental%20Randomized%20Smoothing; https://arxiv.org/abs/2311.09024; DBLP blocked | Proceedings title is identical to the arXiv title: "Fast Certification of Vision-Language Models Using Incremental Randomized Smoothing"; authors in proceedings order Ashutosh Nirala, Ameya Joshi, Soumik Sarkar, Chinmay Hegde (the arXiv listing orders them "A K Nirala, A Joshi, C Hegde, S Sarkar"); event "2024 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), Toronto, ON, Canada, April 9-11, 2024"; pages 252-271; published April 9, 2024; publisher IEEE; DOI 10.1109/SaTML59370.2024.00019 (Crossref stores the DOI lower-cased; DOIs are case-insensitive). arXiv v2 dated 4 Jan 2024, no journal-ref | CONFIRMED |
| `pautov2022smoothed` | https://proceedings.neurips.cc/paper_files/paper/2022/hash/9a07bb7288caaea2ecc4c367188bc6db-Abstract-Conference.html | title; authors (Pautov, Kuznetsova, Tursynbek, Petiushko, Oseledets); NeurIPS 35 (2022) Main Conference Track | CONFIRMED |
| `wu2022retrievalguard` | https://proceedings.mlr.press/v162/wu22o.html | PMLR BibTeX: title; authors; 39th ICML; pages 24266--24279; volume 162; PMLR; 2022 | CONFIRMED |
| `cohen2019smoothing` | https://proceedings.mlr.press/v97/cohen19c.html | PMLR BibTeX: title; authors (Cohen, Rosenfeld, Kolter); 36th ICML; pages 1310--1320; volume 97; PMLR; 2019 | CONFIRMED |
| `zhai2020macer` | https://arxiv.org/abs/2001.02378 (Comments: "Published in ICLR 2020. 20 Pages") | title; authors (Zhai, Dan, He, Zhang, Gong, Ravikumar, Hsieh, Wang); venue ICLR 2020; URL is the arXiv page | CONFIRMED |
| `salman2019provably` | https://proceedings.neurips.cc/paper/2019/hash/3a24b25a7b092a252166a1641ae953e7-Abstract.html; https://proceedings.neurips.cc/paper_files/paper/9307-/bibtex; https://arxiv.org/abs/1906.04584 | title; authors in NeurIPS order (Salman, Li, Razenshteyn, Pengchuan Zhang, Huan Zhang, Bubeck, Yang; the arXiv page orders them Salman, Yang, Li, Zhang, Zhang, Razenshteyn, Bubeck); "Advances in Neural Information Processing Systems 32 (NeurIPS 2019)"; publisher Curran Associates, Inc.; arXiv Comments: "Spotlight at the 33rd Conference on Neural Information Processing Systems (NeurIPS 2019)". Official NeurIPS BibTeX has `pages = {}`, so no pages are given | PARTIAL (pages omitted: not in the official record) |
| `kumar2020certifying` | https://proceedings.neurips.cc/paper/2020/hash/37aa5dfc44dddd0d19d4311e2c7a0240-Abstract.html; https://proceedings.neurips.cc/paper_files/paper/10158-/bibtex; https://arxiv.org/abs/2009.08061 | title; authors (Aounon Kumar, Alexander Levine, Soheil Feizi, Tom Goldstein); NeurIPS 33 (2020); pages 5165--5177; Curran Associates, Inc. | CONFIRMED |
| `carlini2023dds` | https://api.openreview.net/notes/search?term=Certified+Adversarial+Robustness+for+Free (note id JLg5aHHv7j, venue "ICLR 2023 poster", venueid ICLR.cc/2023/Conference); https://arxiv.org/abs/2206.10550 | title "(Certified!!) Adversarial Robustness for Free!"; authors (Nicholas Carlini, Florian Tramer, Krishnamurthy Dj Dvijotham, Leslie Rice, Mingjie Sun, J. Zico Kolter); OpenReview `_bibtex` booktitle "The Eleventh International Conference on Learning Representations", year 2023, url https://openreview.net/forum?id=JLg5aHHv7j | CONFIRMED |
| `hussein2024promptsmooth` | https://api.crossref.org/works/10.1007/978-3-031-72390-2_65 (Springer-deposited); https://citation-needed.springer.com/v2/references/10.1007/978-3-031-72390-2_65?format=bibtex&flavour=citation (Springer citation service); https://arxiv.org/abs/2408.16769 (Comments: "Accepted to MICCAI 2024") | title; authors (Noor Hussein, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar); booktitle "Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024"; series Lecture Notes in Computer Science (Crossref container-title); editors (Linguraru, Dou, Feragen, Giannarou, Glocker, Lekadir, Schnabel); pages 698--708; 2024; Springer Nature Switzerland, Cham; DOI; ISBN 978-3-031-72390-2. LNCS volume number appears in none of the fetched records (Springer chapter page redirects to an IdP), so it is omitted | PARTIAL (LNCS volume omitted) |
| `sato2026hubness` | https://arxiv.org/abs/2609.01103 | title "When Modality Gap Reduction Fails: Prediction-Level Hubness in CLIP"; authors (Shota Sato, Hajime Kiyama, Tosho Hirasawa, Mamoru Komachi); v1 Tue, 1 Sep 2026; categories cs.CL, cs.CV; Comments field reads "Accpeted to EMNLP 2026 Main Conference" (sic). Kept as an arXiv entry with the venue in `note` because no EMNLP 2026 proceedings record exists yet | CONFIRMED |
| `radford2021clip` | https://proceedings.mlr.press/v139/radford21a.html | PMLR BibTeX: title; 12 authors (Radford ... Sutskever); 38th ICML; pages 8748--8763; volume 139; PMLR; 2021 | CONFIRMED |
| `cherti2023openclip` | https://openaccess.thecvf.com/content/CVPR2023/html/Cherti_Reproducible_Scaling_Laws_for_Contrastive_Language-Image_Learning_CVPR_2023_paper.html (curl, HTTP 200); https://api.crossref.org/works?query.bibliographic=Reproducible+Scaling+Laws+for+Contrastive+Language-Image+Learning (IEEE record) | CVF bibref: authors (Cherti, Beaumont, Wightman, Wortsman, Ilharco, Gordon, Schuhmann, Schmidt, Jitsev); booktitle Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); June 2023; pages 2818-2829. IEEE/Crossref: "2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", pages 2818-2829, DOI 10.1109/CVPR52729.2023.00276 | CONFIRMED |
| `schuhmann2022laion5b` | https://proceedings.neurips.cc/paper_files/paper/2022/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html; https://proceedings.neurips.cc/paper_files/paper/17353-/bibtex; https://arxiv.org/abs/2210.08402 | title; 16 authors (Schuhmann ... Jitsev); NeurIPS 35 (2022) Datasets and Benchmarks Track; pages 25278--25294; Curran Associates, Inc.; DOI 10.52202/068431-1833 (from the official NeurIPS BibTeX); arXiv Comments: "36th Conference on Neural Information Processing Systems (NeurIPS 2022), Track on Datasets and Benchmarks" | CONFIRMED |
| `krizhevsky2009cifar` | https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf (301 -> https://cave.cs.toronto.edu/kriz/learning-features-2009-TR.pdf; PDF fetched, page 1 extracted with pdftotext); https://cave.cs.toronto.edu/kriz/cifar.html | Page 1 prints exactly: "Learning Multiple Layers of Features from Tiny Images / Alex Krizhevsky / April 8, 2009". The official CIFAR page cites it as "Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009" (tech report). The PDF text contains no affiliation line and no "University of Toronto" string, so `institution` is omitted | PARTIAL (institution omitted) |
| `helber2019eurosat` | https://api.crossref.org/works/10.1109/JSTARS.2019.2918242 (IEEE-deposited); IEEE Xplore page returned empty content | title "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"; authors (Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth); IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing; vol 12, no 7, pp 2217-2226; July 2019; DOI | CONFIRMED (via Crossref) |
| `clopper1934` | https://academic.oup.com/biomet/article/26/4/404/291538 (OUP page, HTTP 200); https://api.crossref.org/works/10.1093/biomet/26.4.404 | title "The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial"; authors C. J. Clopper and E. S. Pearson; Biometrika 26(4):404-413; December 1934; DOI | CONFIRMED |
| `lakens2017equivalence` | https://api.crossref.org/works/10.1177/1948550617697177 (SAGE-deposited); SAGE page HTTP 403 | title "Equivalence Tests" / subtitle "A Practical Primer for t Tests, Correlations, and Meta-Analyses"; author Daniël Lakens; Social Psychological and Personality Science 8(4):355-362; 2017; DOI | CONFIRMED (via Crossref) |
| `kuchibhotla2022postselection` | https://api.crossref.org/works/10.1146/annurev-statistics-100421-044639 (Annual Reviews-deposited); AR page HTTP 403 | title "Post-Selection Inference"; authors (Arun K. Kuchibhotla, John E. Kolassa, Todd A. Kuffner); Annual Review of Statistics and Its Application 9(1):505-527; 2022; DOI | CONFIRMED (via Crossref) |
| `lecuyer2019pixeldp` | https://api.crossref.org/works/10.1109/SP.2019.00044 (IEEE-deposited); IEEE Xplore page returned empty content | title; authors (Mathias Lecuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, Suman Jana); "2019 IEEE Symposium on Security and Privacy (SP)"; pages 656-672; May 2019; IEEE; DOI | CONFIRMED (via Crossref) |
| `li2019certified` | https://proceedings.neurips.cc/paper/2019/hash/335cd1b90bfa4ee70b39d08a4ae0cf2d-Abstract.html; https://proceedings.neurips.cc/paper_files/paper/9143-/bibtex; https://arxiv.org/abs/1809.03113 (Comments: "NIPS 2019") | title; authors (Bai Li, Changyou Chen, Wenlin Wang, Lawrence Carin); NeurIPS 32 (2019); Curran Associates, Inc. Official NeurIPS BibTeX has `pages = {}`, so no pages are given | PARTIAL (pages omitted: not in the official record) |
| `yang2020randomized` | https://proceedings.mlr.press/v119/yang20c.html | PMLR BibTeX: title; authors (Greg Yang, Tony Duan, J. Edward Hu, Hadi Salman, Ilya Razenshteyn, Jerry Li); 37th ICML; pages 10693--10705; volume 119; PMLR; 2020 | CONFIRMED |
| `zhou2022coop` | https://api.crossref.org/works/10.1007/s11263-022-01653-1 (Springer-deposited); Springer page redirects to an IdP | title "Learning to Prompt for Vision-Language Models"; authors (Kaiyang Zhou, Jingkang Yang, Chen Change Loy, Ziwei Liu); International Journal of Computer Vision 130(9):2337-2348; 2022; DOI | CONFIRMED (via Crossref) |
| `nosek2018preregistration` | https://api.crossref.org/works/10.1073/pnas.1708274114 (PNAS-deposited); PNAS page HTTP 403 | title "The preregistration revolution"; authors (Brian A. Nosek, Charles R. Ebersole, Alexander C. DeHaven, David T. Mellor); Proceedings of the National Academy of Sciences 115(11):2600-2606; 2018; DOI | CONFIRMED (via Crossref) |
| `hu2022lora` | https://api.openreview.net/notes/search?term=LoRA+Low-Rank+Adaptation+of+Large+Language+Models (note id nZeVKeeFYf9, venue "ICLR 2022 Poster", venueid ICLR.cc/2022/Conference, `_bibtex` booktitle "International Conference on Learning Representations", year 2022, url https://openreview.net/forum?id=nZeVKeeFYf9); https://arxiv.org/abs/2106.09685 | title; 8 authors (Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen); venue; URL | CONFIRMED |

### 1.1 Summary of statuses

- CONFIRMED: 31 entries.
- PARTIAL: 5 entries
  - `salman2019provably` (pages omitted; official NeurIPS BibTeX has an empty pages field; author order follows the NeurIPS proceedings, which differs from arXiv)
  - `li2019certified` (pages omitted; official NeurIPS BibTeX has an empty pages field)
  - `hussein2024promptsmooth` (LNCS volume number omitted; not in Crossref or the Springer citation export)
  - `krizhevsky2009cifar` (institution omitted; the PDF prints only title, author, and "April 8, 2009")
  - `schrodi2025twoeffects` (copied entry; OpenReview forum id uAFHCZRmXk could not be re-fetched, title/authors/venue confirmed via arXiv)
- UNVERIFIED: none.

### 1.2 Discrepancies and decisions worth knowing

- `nirala2024ovc`: the SaTML 2024 proceedings title is identical to the arXiv
  title. Author order in the proceedings is Nirala, Joshi, Sarkar, Hegde; the
  arXiv listing has Nirala, Joshi, Hegde, Sarkar. The bib uses the proceedings
  order.
- `salman2019provably`: NeurIPS proceedings author order (Salman, Li,
  Razenshteyn, P. Zhang, H. Zhang, Bubeck, Yang) differs from the arXiv order
  (Salman, Yang, Li, P. Zhang, H. Zhang, Razenshteyn, Bubeck). The bib uses the
  proceedings order.
- `sato2026hubness`: arXiv Comments say "Accpeted to EMNLP 2026 Main
  Conference" (typo in the source). Because no proceedings record exists yet,
  the entry stays an arXiv preprint entry and carries the acceptance in `note`.
  Re-check for an ACL Anthology record before camera-ready.
- `chowers2026bug`: the ICLR-template entry was correct; `month = {June}` was
  added from the CVF bibref. The CVF page carries no DOI.
- `cherti2023openclip`: CVF gives pages 2818-2829; the IEEE DOI
  10.1109/CVPR52729.2023.00276 (same pages) is included from Crossref.
- No entry copied from the ICLR template was altered except `nirala2024ovc`
  (replaced) and `chowers2026bug` (month added).

## 2. SaTML 2027 Call for Papers requirements (verbatim)

Sources: https://satml.org/call-for-papers/ ("Last updated: Sep 22, 2026") and
https://satml.org/call-for-papers/checklist/ ("Last updated: Sep 23, 2026").
Both pages were downloaded with curl on 2026-09-25 and the text below was
extracted from the HTML without paraphrase.

### (a) "LLM usage considerations" editorial statement

Call for Papers, "Submission information":

> ⚠️Usage of AI/LLM: If any LLM was used, an "LLM usage considerations" section is required, placed after the Open Science section. See the dedicated section below ("Usage of LLMs") for what it must contain.

Call for Papers, section "Usage of LLMs" (complete):

> Authors are permitted to use LLMs when preparing their paper. However, while the conference does not ban authors from using LLMs or researching their security and privacy properties, authors must (a) carefully consider their decision to use LLMs and (b) are required to disclose and motivate the use of LLMs in their submission. If the authors choose to use LLMs in their work, they must use a separate and well-marked section titled "LLM usage considerations" before the references (after Open Science), to make the relevant disclosures. This section will not count towards the page limit.
>
> We ask that authors adhere to three key criteria with regards to their use of LLMs in the scientific process:
>
> Accountability and Correctness: Human authors are ultimately responsible for all submitted content and results, ensuring their accuracy, originality, and integrity. Works submitted should constitute self-respecting work that respects the importance of scientific inquiry and integrity and respects the time and energy of reviewers. Authors are responsible for the thoroughness of their literature review and must determine relevant prior work and cite it to ensure proper credit. Upon submitting to SaTML 2027, authors acknowledge that they have reviewed all content that was generated by AI as if they had written it themselves, including text, code, experimental data, and references. Any violation to this policy which results in fabrications or hallucinations, including non-existing references or incorrect authors, invented claims, and falsified results is considered academic misconduct and might lead to the paper being desk rejected or other sanctions. Authors are encouraged to evaluate their submissions using the same open-source tools that we will use (e.g., https://hallucinator.science/). If the authors have used LLMs to improve their writing, they should state: 'LLMs were used for editorial purposes in this manuscript, and all outputs were inspected by the authors to ensure accuracy and originality.'
>
> Transparency: Second, authors should carefully reason about the implications of using LLMs in their work. If LLMs are integral to the paper's methodology, their use should be explicitly detailed. Any idea generated by an LLM should be independently developed and validated by the authors. Furthermore, authors must elaborate on how they handled limitations introduced in their work by their use of LLMs. Such limitations could for instance include difficulties to obtain results that are reproducible when the LLM used is not open sourced.
>
> Responsibility: Third, authors should take care to develop LLMs (and ML models in general) responsibly. Any data collection towards training models should take into account relevant ethical considerations such as consent and data holder rights, including intellectual property. Authors also have to justify the need for the environmental footprint of their experiments to achieve their goals and support their methodology. We recognize calculating such a footprint is a technical challenge in itself. We refer the authors to the work of Lacoste et al. but welcome to hear any other good references (pcchairs@satml.org). We emphasize that the goal here is not to calculate the exact footprint but rather explain experimental choices made as part of the scientific process (e.g., why was an LLM necessary, why was a particular model size selected, how the authors minimized the volume of queries made, which hardware was used to run experiments).
>
> Failure to comply with these rules is grounds for desk rejection without further review of the submission. We note that generative AI technology is rapidly evolving. Authors are encouraged to reach out proactively to the PC chairs should they face uncertainties about the above rules or how they apply to their research.

The required editorial-use sentence, exactly as the CfP gives it (the page
sets it in single quotation marks):

> LLMs were used for editorial purposes in this manuscript, and all outputs were inspected by the authors to ensure accuracy and originality.

Checklist page, "By the paper submission deadline":

> "LLM usage considerations" section (after open science, before references) if any LLM was used, including the required editorial-use statement. If LLMs are part of the method, detail their role, limitations, and compute justification.

> The Open Science, LLM usage considerations, and Ethical Considerations sections do not count towards the page limit.

Checklist page, "Desk rejection" (relevant item):

> violations of the Open Science or LLM policies, including non-existent references

### (b) Open Science section requirement

Call for Papers, "Submission information":

> Open Science: All submissions must include an "Open Science" section immediately before the references, which does not count towards the page limit. In it, the authors should describe which artifacts they are releasing, or explain why sharing is not possible. The section is required for every submission category: Position papers, for instance, often have nothing to release, but the section is still compulsory and should say so. We require all authors to share their artifacts within 3 days after the submission deadline, using a fully-anonymized repository (for example, https://anonymous.4open.science/). After this deadline, the artifact(s) needs to be accessible throughout the review process, and should not be edited. Any violation of the Open Science policy may result in desk rejection. Authors of accepted papers will be required to share their artifacts on https://zenodo.org (see "Important Dates"), and acceptance is conditional on their availability (unless a valid reason is provided). Chairs and reviewers may check that the artifacts are consistent with the claims made in the paper.

Call for Papers, "Important dates" (artifact-related lines):

> Anonymized artifact(s) updated by: Fri, Oct 2, 2026

> Final artifacts (of accepted papers) on Zenodo due by: Thu, Jan 14, 2027

> Final artifacts (of accepted revised papers) on Zenodo due by: camera-ready deadline

Call for Papers, "Submission decisions":

> Note: Acceptance is conditional on artifact availability on zenodo.org (see "Open Science"), or a valid justification for why this is not possible.

Checklist page, "By the paper submission deadline":

> "Open Science" section right before references: what you release, or why you cannot. Required for every category, including Position papers.

> Anonymous repository link (e.g. anonymous.4open.science) submitted in HotCRP. If the paper has no code, write "See Open Science section": every paper needs that section, even a purely theoretical one.

Checklist page, "Artifacts, Fri 2 Oct 2026":

> Last chance to update the anonymous artifact repository linked from your HotCRP submission (e.g. anonymous.4open.science). The artifact must remain accessible and unedited for the rest of the review process.

Checklist page, "If accepted":

> Artifacts on Zenodo, by Thu 14 Jan 2027 (for accepted papers), or the camera-ready deadline (for papers invited to revise, and later accepted).

### (c) Anonymity rules for artifacts

Call for Papers, "Submission information":

> Double-blind review: SaTML follows a double-blind reviewing process. Submitted papers must be properly anonymized! They must (a) omit any reference to the authors' names or their institutions, and (b) cite the authors' own related work in the third person. It is important, however, to ensure that efforts to maintain anonymity do not compromise the quality of the submission or complicate the review process. Essential background references, for example, should not be omitted or anonymized. Even if a paper is anonymised, it should not make statements such as "our artifacts are already available as open source tools", as it may deanonymize them. In general, any material referenced in the paper should be anonymized, and de-anonymization through additional material such as artifacts are grounds for rejection. Please see this double-blind FAQ for the answers to many common concerns about double-blind reviewing.

> Preprint/disclosure policy: Authors may choose to give talks about their work, post a preprint of the paper online, and disclose security vulnerabilities to vendors. However, authors should take care of avoiding behaviors that intentionally aim to inform reviewers of their identity (e.g., publicly advertising their work on social media), as such behaviors may lead to desk rejection of the paper. If in doubt, the authors should contact the PC chairs at pcchairs@satml.org.

Checklist page, "By the paper submission deadline":

> Anonymized: no names or affiliations, own work cited in third person, no hints that artifacts are already public, all linked material anonymous. Keep essential background citations.

Checklist page, "Desk rejection" (relevant item):

> broken anonymity, or self-promotion that reveals author identity

### (d) "hallucinator.science" instruction

Call for Papers, inside the "Accountability and Correctness" paragraph quoted in (a):

> Authors are encouraged to evaluate their submissions using the same open-source tools that we will use (e.g., https://hallucinator.science/).

Checklist page, "By the paper submission deadline":

> Verify every reference and result yourself. Fabrications count as misconduct. Self-check with hallucinator.science.

### (e) Other formatting facts recorded while on the pages

Call for Papers:

> Submission template: Submissions must be a PDF file in two-column IEEE proceedings style. That is, authors must use \documentclass[conference]{IEEEtran} when preparing their paper, with the default 10pt font size and page geometry. Using a different template, or modifying font size, margins, or spacing to fit more content, is grounds for desk rejection. The number of allowed pages for a submission depends on the submission category, see above.

> Ethical Considerations: Each paper may optionally include an "Ethical Considerations" section, placed immediately before the references. This section does not count towards the page limit. In this section, the authors may discuss if they believe the work poses any ethical risk and the steps that are taken to mitigate such risk, also referring to ethics frameworks such as those offered by the Menlo report. If the authors believe that their work does not pose any ethical considerations, this section is not necessary.

Checklist page, "Before writing":

> Category: Research 12pp body text; SoK 12pp body text, title starts "SoK:"; Position 5 to 12pp body text, title starts "Position:". SoK and Position also need a 300-word "New Insights" field in HotCRP; the field is mandatory, so research papers enter "N/A". References and appendices unlimited.

> Format: PDF, \documentclass[conference]{IEEEtran}, 10pt, no change in geometry/template.

## 3. Syntax check (bibtex-free)

Command: `python bibcheck.py paper/satml2027/references.bib` (script kept in
the session scratchpad; it parses every `@type{key, ...}` block, tracks brace
depth while honouring `\{`/`\}`, extracts each field, and reports unterminated
entries, per-field brace imbalance, duplicate keys, missing required fields
per entry type, empty fields, and non-ASCII characters).

Result on 2026-09-25:

- Entries parsed: 36 (matches the 36 `@` lines in the file).
- Global brace balance: OK. Per-field brace balance: OK.
- Duplicate keys: none.
- Missing required fields (author/title/journal-or-booktitle/year): none.
- Empty fields: none.
- Non-ASCII characters: none (the only accented name, Lakens, is written as
  `Dani{\"e}l`).

Keys (36, in file order):

```
xia2026feature, maurer2009empirical, chowers2026bug, liang2022mind,
schrodi2025twoeffects, fahim2024contrastive, li2025closing,
dong2025stabilizing, zhu2025cola, dong2025pathsimplices, dong2025subspaces,
schlarmann2024robustclip, nirala2024ovc, pautov2022smoothed,
wu2022retrievalguard, cohen2019smoothing, zhai2020macer,
salman2019provably, kumar2020certifying, carlini2023dds,
hussein2024promptsmooth, sato2026hubness, radford2021clip,
cherti2023openclip, schuhmann2022laion5b, krizhevsky2009cifar,
helber2019eurosat, clopper1934, lakens2017equivalence,
kuchibhotla2022postselection, lecuyer2019pixeldp, li2019certified,
yang2020randomized, zhou2022coop, nosek2018preregistration, hu2022lora
```

### 3.1 Required-key check

A second script compared the key list against (i) the 22 keys named in the
task and (ii) the 17 keys of `paper/iclr2027-template/iclr2027/references.bib`:

- required new/verified keys missing: none (22/22 present);
- original ICLR-template keys missing: none (17/17 present);
- union of the two sets = 36 = number of entries in the file; duplicates: none.

### 3.2 Real `bibtex` run with `IEEEtran.bst`

A throwaway document was compiled in the session scratchpad (not in the
repository) with TeX Live 2025:

```
\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc} \usepackage[utf8]{inputenc} \usepackage{url}
... \nocite{*} \bibliographystyle{IEEEtran} \bibliography{references}
```

Sequence `pdflatex -> bibtex -> pdflatex -> pdflatex`, all exit code 0.
BibTeX 0.99d with `IEEEtran.bst` version 1.14 (2015/08/26) reported:

```
Database file #1: references.bib
Warning--missing institution in krizhevsky2009cifar
Done.
You've used 36 entries,
(There was 1 warning)
```

- All 36 entries were accepted and typeset; no "undefined citation" and no
  error in the LaTeX log; the test PDF has 2 pages.
- The single warning is the deliberate omission recorded under
  `krizhevsky2009cifar` (PARTIAL): `IEEEtran.bst` treats `institution` as a
  required field of `@techreport`. The entry still renders as
  `A. Krizhevsky, "Learning multiple layers of features from tiny images,"
  Tech. Rep., April 2009. [Online]. Available: <url>` (the earlier
  `note = {Technical report}` was dropped because IEEEtran already prints
  "Tech. Rep."). Two clean resolutions, for the authors to choose: add
  `institution = {University of Toronto}` on the strength of the report being
  hosted on the University of Toronto CS server and linked from the official
  CIFAR page (the PDF text itself carries no affiliation), or change the entry
  to `@misc` with `howpublished = {Technical report}`. Neither was applied,
  to avoid writing an unconfirmed field.

## 4. Follow-ups before submission

1. Run the manuscript through https://hallucinator.science/ as the CfP asks;
   the five PARTIAL entries above are the ones to watch.
2. If a NeurIPS 2019 page range is wanted for `salman2019provably` or
   `li2019certified`, confirm it on an official page first (OpenAlex lists
   11289-11300 and 9459-9469 respectively, but that was not confirmed on an
   official source and is deliberately not in the bib).
3. Re-check `sato2026hubness` for an ACL Anthology / EMNLP 2026 proceedings
   record before camera-ready.
4. RESOLVED (2026-09-27, D-137): a real `bibtex` run under
   `\documentclass[conference]{IEEEtran}` with `\bibliographystyle{IEEEtran}`
   resolves every citation (0 undefined); see section 5. When this item was
   written, the check above was a structural check only.

## 5. `references_extra.bib` (5 entries; added 2026-09-25, verified 2026-09-27)

Trigger: the two reviews of reviewer bundle V4 (2026-09-27) found that manuscript review V3 omitted this file and that
this log did not cover it (D-137). Each entry was re-checked field by field against the official record with curl
(browser user agent), under the method of the header. Three corrections were made: `mohapatra2021hidden` had followed
the arXiv author order, `anani2024adaptive` and `mohapatra2021hidden` gained the pages and PMLR URLs of the official
records, and the unconfirmed `address` of `westfall1993resampling` was removed.

| Key | Source(s) | Fields confirmed | Status |
|---|---|---|---|
| `mohapatra2021hidden` | https://proceedings.mlr.press/v130/mohapatra21a.html (PMLR BibTeX `pmlr-v130-mohapatra21a`); https://arxiv.org/abs/2003.01249 | PMLR: title "Hidden Cost of Randomized Smoothing"; authors in proceedings order Mohapatra, Ko, Weng (PMLR form "Lily"), Chen, Liu, Daniel; booktitle "Proceedings of The 24th International Conference on Artificial Intelligence and Statistics"; pages 4033--4041; volume 130; PMLR; 2021. The arXiv listing orders the authors Mohapatra, Ko, Tsui-Wei Weng, Liu, Chen, Daniel; the entry previously followed that order although it cites the proceedings. Corrected to the PMLR order and name forms; pages and PMLR URL added | CONFIRMED (corrected 2026-09-27) |
| `anani2024adaptive` | https://proceedings.mlr.press/v235/anani24a.html (PMLR BibTeX `pmlr-v235-anani24a`) | title; authors (Alaa Anani, Tobias Lorenz, Bernt Schiele, Mario Fritz); booktitle "Proceedings of the 41st International Conference on Machine Learning"; pages 1531--1556; volume 235; PMLR; 2024. Pages and PMLR URL added (the entry previously pointed to arXiv 2402.08400) | CONFIRMED (completed 2026-09-27) |
| `card2020power` | https://aclanthology.org/2020.emnlp-main.745.bib (official ACL Anthology BibTeX); https://aclanthology.org/2020.emnlp-main.745/ | title "With Little Power Comes Great Responsibility"; authors (Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, Dan Jurafsky); booktitle "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)"; pages 9263--9274; 2020; publisher Association for Computational Linguistics; DOI 10.18653/v1/2020.emnlp-main.745 (added); URL | CONFIRMED |
| `bertinetto2021preregistration` | https://proceedings.mlr.press/v148/ (PMLR volume page) | "NeurIPS 2020 Workshop on Pre-registration in Machine Learning", held virtually on 11 December 2020, published as Volume 148 of the Proceedings of Machine Learning Research on 08 July 2021; volume editors Luca Bertinetto, João F. Henriques, Samuel Albanie, Michela Paganini, Gül Varol. The entry's title prefixes "Proceedings of the", as PMLR proceedings are cited | CONFIRMED |
| `westfall1993resampling` | https://www.wiley.com/en-us/Resampling+Based+Multiple+Testing%3A+Examples+and+Methods+for+p+Value+Adjustment-p-9780471557616 (publisher page); Crossref JSTOR records of 1993-1994 book reviews (e.g. 10.2307/1270279) | Wiley: title "Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment"; authors Peter H. Westfall and S. Stanley Young; January 1993; ISBN 978-0-471-55761-6 (added); series Wiley Series in Probability and Statistics (added). The former `address = {New York}` appears on no fetched official record and was removed | CONFIRMED (corrected 2026-09-27) |

A real `bibtex` run under `\documentclass[conference]{IEEEtran}` with `\bibliographystyle{IEEEtran}` now resolves every
citation of the manuscript (0 undefined citations; see D-137), which settles follow-up 4 of section 4.


## 6. Entries added after the reviews of manuscript V11 (2026-09-29, D-159)

Three entries were added to `references_extra.bib` and checked on 2026-09-29 against the official records below. No
field was written that does not appear in the named source.

| Key | Source | Fields confirmed | Status |
| --- | --- | --- | --- |
| `seferis2025randomized` | https://aclanthology.org/2025.emnlp-main.1396/ and its official BibTeX export https://aclanthology.org/2025.emnlp-main.1396.bib | title; five authors (Seferis, Wu, Kollias, Bensalem, Cheng); booktitle "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing"; pages 27468--27478 (the official export; a search-engine snippet showed 27456--27466 and was not used); year 2025; address Suzhou, China; publisher ACL; DOI 10.18653/v1/2025.emnlp-main.1396. The abstract (mapping generative outputs to an oracle classification task) supports the sentence in Related Work. | CONFIRMED |
| `levi2025double` | https://proceedings.mlr.press/v267/levi25b.html (PMLR page with its BibTeX) | title "The Double-Ellipsoid Geometry of CLIP"; authors Meir Yossef Levi, Guy Gilboa; ICML 2025 proceedings, PMLR volume 267, pages 33999--34019, year 2025, publisher PMLR. The abstract ("text and image reside on linearly separable ellipsoid shells, not centered at the origin") supports the sentence in Related Work. | CONFIRMED |
| `howard2019imagenette` | https://github.com/fastai/imagenette (README: "Imagenette is a subset of 10 easily classified classes from Imagenet"; created by Jeremy Howard of fast.ai); the dataset citation published in the TensorFlow Datasets catalog, https://www.tensorflow.org/datasets/catalog/imagenette (title, author, year 2019, month March, publisher GitHub, URL) | title, author, year, month, URL; entered as `@misc` because `IEEEtran.bst` has no software type | CONFIRMED |
