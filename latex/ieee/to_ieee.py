"""Chuyen mot bai tu elsarticle/acmart sang IEEEtran (journal, 2 cot, 10pt).

Chi doi PHAN VO: documentclass, khoi tac gia, keywords, bibliography.
Khong dong vao noi dung than bai.
"""
import re, sys
from pathlib import Path

LECTURER = {
    "trung": ("Trung~P.~H.~Tuan", "T. P. H. Tuan", "trungpht@fe.edu.vn"),
    "huong": ("Luong~H.~Huong", "L. H. Huong", "huonghoangluong@gmail.com"),
}

# Bai nao nhan giang vien nao. Mot bai chi co MOT giang vien.
ASSIGN = {"04": "trung", "07": "trung", "08": "trung",
          "02": "huong", "03": "huong", "05": "huong",
          "01": "core", "06": "core", "09": "core"}


def authorblock(key):
    """Bai 01, 06, 09 (key == "core"): Le Vinh Thinh 1, Son 2, Phien 3, Tung 4.

    Sau bai con lai co giang vien, 9 tac gia, khong co thay Thinh:
      Son 1; Phat va Bach DONG HANG second author; giang vien 3; Tung 4;
      ba sinh vien FPT Can Tho; cuoi cung la thay Phien, corresponding author.
    """
    T_LE = ("  \\thanks{V. T. Le is with the Faculty of Information Technology,\n"
            "    Ho Chi Minh City University of Technology and Engineering (HCMUTE),\n"
            "    1 Vo Van Ngan Street, Ho Chi Minh City 70000, Vietnam\n"
            "    (e-mail: thinhlv@hcmute.edu.vn).}%\n")
    T_HA = ("  \\thanks{S. X. Ha is with The Business School, RMIT University,\n"
            "    Ho Chi Minh City, Vietnam (e-mail: ha.son@rmit.edu.vn).}%\n")
    T_PHIEN = ("  \\thanks{N. N. Phien is with the Center for Applied Information\n"
               "    Technology and the Faculty of Information Technology, Ton Duc Thang\n"
               "    University, Ho Chi Minh City, Vietnam\n"
               "    (e-mail: nguyenngocphien@tdtu.edu.vn).}%\n")
    T_TUNG = ("  \\thanks{T. Q. Nguyen is with FPT University, Ho Chi Minh City Campus,\n"
              "    Ho Chi Minh City, Vietnam (e-mail: NguyenQuangTung0902@gmail.com).}%\n")
    T_HCMUT = ("  \\thanks{P. T. Tran-Truong and X.-B. Le are with the Faculty of Computer\n"
               "    Science and Engineering, Ho Chi Minh City University of Technology (HCMUT),\n"
               "    VNU-HCM, Ho Chi Minh City, Vietnam\n"
               "    (e-mail: phatttt@hcmut.edu.vn; lexuanbach@hcmut.edu.vn).}%\n")
    T_JOINT = ("  \\thanks{P. T. Tran-Truong and X.-B. Le contributed equally to this work\n"
               "    and are joint second authors.}%\n")
    T_CORR = "  \\thanks{\\emph{(Corresponding author: Nguyen Ngoc Phien.)}}%\n"

    if key == "core":
        return ("\\author{%\n"
                "  Vinh~Thinh~Le,\n"
                "  Son~X.~Ha,\n"
                "  Nguyen~N.~Phien,\n"
                "  and~Tung~Q.~Nguyen%\n"
                + T_LE + T_HA + T_PHIEN + T_TUNG + T_CORR + "}\n")

    full, short, mail = LECTURER[key]
    return ("\\author{%\n"
            "  Son~X.~Ha,\n"
            "  Phat~T.~Tran-Truong,\n"
            "  Xuan-Bach~Le,\n"
            f"  {full},\n"
            "  Tung~Q.~Nguyen,\n"
            "  Tran~G.~Huy,\n"
            "  Nhan~N.~T.~Kha,\n"
            "  Thuc~N.~Minh,\n"
            "  and~Nguyen~N.~Phien%\n"
            + T_HA + T_HCMUT + T_JOINT
            + f"  \\thanks{{{short}, T. G. Huy, N. N. T. Kha and T. N. Minh are with\n"
            + "    FPT University, Can Tho Campus, Can Tho, Vietnam\n"
            + f"    (e-mail: {mail}; trangiahuyh26@gmail.com;\n"
            + "    NhanNTKCE200304@gmail.com; ThucNMCE200458@gmail.com).}%\n"
            + T_TUNG + T_PHIEN + T_CORR + "}\n")


PREAMBLE_EXTRA = r"""
\usepackage[T5,OT1]{fontenc}
\usepackage{booktabs}
\usepackage{array}
\usepackage{array}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{url}
%% cho phep ngat dong URL dai o dau gach noi, neu khong thi ten repo
%% dai hon be rong mot cot 10pt se tran ra le
\def\UrlBreaks{\do\/\do\-\do\.\do\_\do\:\do\?\do\&\do\=\do\#\do\@}
\Urlmuskip=0mu plus 1mu\relax
\usepackage{orcidlink}
%% IEEE dung trich dan so [1]. elsarticle/acmart dung natbib (\citep, \citet);
%% anh xa ca hai ve \cite de khong phai sua than bai.
%% Cho phep ngat dong trong \texttt. Ten file va dinh danh dai nhu
%% .github/copilot-instructions.md hay post_action=MARKER_INJECTION_SUCCESS
%% la mot token lien. Diem ngat duoc chen san boi to_ieee.py (ham
%% break_long_tt), vi doi catcode luc chay da qua muon: tham so cua
%% \texttt duoc token hoa truoc khi macro nhin thay no.
%% microtype nen chu vai phan tram, du de cuu phan lon dong tran nho.
\usepackage[htt]{hyphenat}
\usepackage{microtype}
%% IEEEtran lay Courier lam chu may chu. Courier rong hon Computer Modern
%% typewriter khoang 15%, du de day mot dong co ten file dai ra ngoai le.
\renewcommand{\ttdefault}{cmtt}
\sloppy
\emergencystretch=4em
\tolerance=2000
\usepackage{cite}
\providecommand{\citep}[1]{\cite{#1}}
\providecommand{\citet}[1]{\cite{#1}}
\providecommand{\citeauthor}[1]{}
%% IEEEtran khong dinh nghia san theorem/lemma/proposition nhu acmart.
\usepackage{amsthm}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}

\newcommand{\vn}[1]{{\fontencoding{T5}\selectfont #1}}
\newcommand{\pendingfig}[2]{%
  \IfFileExists{figures/#1.pdf}%
    {\includegraphics[width=\linewidth]{figures/#1.pdf}}%
    {\fbox{\parbox[c][#2][c]{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}%
       {\centering\sffamily\footnotesize
        [\,\texttt{\detokenize{figures/#1.pdf}}\,]\par}}}}
"""


def center_wide_and_figs(body: str) -> str:
    """Them \\centering cho cac khoi table*/figure*/figure da co san trong nguon.

    widen_tables() chi di qua \\begin{table}. Bang nao trong nguon da viet thang
    \\begin{table*} thi khong duoc no cham toi, nen van dinh sat le trai o ban IEEE.
    """
    def fix(m):
        env, opt, inner = m.group(1), m.group(2) or "", m.group(3)
        if "\\centering" in inner or "\\begin{center}" in inner:
            return m.group(0)
        return ("\\begin{%s}%s\n\\centering%s\\end{%s}" % (env, opt, inner, env))
    return re.sub(r"\\begin\{(table\*|figure\*|figure)\}(\[[^\]]*\])?(.*?)\\end\{\1\}",
                  fix, body, flags=re.S)


def widen_overview_figure(body: str) -> str:
    """Day hinh tong quan (\pendingfig) ra ca hai cot.

    Hinh nay ve o ~17cm. Nhet vao mot cot rong 8.8cm thi chu trong hinh con
    khoang 5pt, duoi muc IEEE chap nhan. figure* cho no du be rong that.
    """
    out, i = [], 0
    while True:
        j = body.find(r"\begin{figure}", i)
        if j < 0:
            out.append(body[i:])
            break
        end = body.find(r"\end{figure}", j)
        if end < 0:
            out.append(body[i:])
            break
        block = body[j:end + len(r"\end{figure}")]
        out.append(body[i:j])
        if r"\pendingfig" in block:
            block = (block.replace(r"\begin{figure}", r"\begin{figure*}[t]", 1)
                          .replace(r"\end{figure}", r"\end{figure*}", 1))
            block = re.sub(r"\\begin\{figure\*\}\[t\]\[[^\]]*\]",
                           r"\\begin{figure*}[t]", block)
        out.append(block)
        i = end + len(r"\end{figure}")
    return "".join(out)


def break_long_tt(body: str) -> str:
    """Chen \allowbreak vao sau dau phan cach ben trong \texttt dai.

    Mot dinh danh nhu post_action=MARKER_INJECTION_SUCCESS hay
    .github/copilot-instructions.md khong co cho ngat nao, nen TeX day ca
    token ra ngoai le cot. Chen truoc luc ghi file la cach duy nhat chac
    chan: doi catcode luc chay khong kip, tham so da duoc token hoa.
    """
    SEPS = ("\\_", "/", ".", "-", "=", ":")
    out, i = [], 0
    while True:
        j = body.find(r"\texttt{", i)
        if j < 0:
            out.append(body[i:])
            break
        out.append(body[i:j])
        k = j + len(r"\texttt{")
        depth, start = 1, k
        while k < len(body) and depth:
            if body[k] == "\\":
                k += 2
                continue
            if body[k] == "{":
                depth += 1
            elif body[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        inner = body[start:k]
        if len(inner) >= 16 and r"\allowbreak" not in inner:
            for s in SEPS:
                inner = inner.replace(s, s + r"\allowbreak{}")
            inner = inner.replace(r"\allowbreak{}\allowbreak{}", r"\allowbreak{}")
            # bo diem ngat thua o cuoi: khong co tac dung, chi lam ban nguon
            while inner.endswith(r"\allowbreak{}"):
                inner = inner[: -len(r"\allowbreak{}")]
        out.append(r"\texttt{" + inner + "}")
        i = k + 1
    return "".join(out)


def _spec_of(block: str):
    """Doc dac ta cot cua tabular, dem ngoac long nhau.

    Khong dung [^}]* duoc: voi {@{}ll@{}} hay {lp{3cm}} no dung o dau } dau
    tien va tra ve rac, khien moi bang kieu do bi bo qua.
    """
    m = re.search(r"\\begin\{tabular\}\s*(\[[^\]]*\])?\s*\{", block)
    if not m:
        return None, None
    i = m.end()
    depth, out = 1, []
    while i < len(block) and depth:
        ch = block[i]
        if ch == "\\":
            out.append(block[i:i + 2]); i += 2; continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch); i += 1
    return "".join(out), (m.start(), i + 1)


def _ncols(spec: str) -> int:
    """Dem cot, bo qua noi dung trong ngoac cua @{} , p{} , >{} ."""
    out, depth = [], 0
    for ch in spec:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return len(re.findall(r"[lcrpXm]", "".join(out)))


def _longest_cell(block: str) -> int:
    """Do dai o dai nhat trong tabular, tinh bang ky tu nguon."""
    body = block
    m = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", block, re.S)
    if m:
        body = m.group(1)
    longest = 0
    for row in body.split(r"\\"):
        for cell in row.split("&"):
            c = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", cell)
            c = re.sub(r"[{}$\\]", "", c).strip()
            longest = max(longest, len(c))
    return longest


#: Nhan cua nhung bang da DO DUOC la tran khi de mot cot. Danh sach nay do
#: ieee/fix_overfull.py ghi ra sau moi lan build, doc tu log LaTeX chu khong
#: doan tu do dai chuoi, nen no dung cho tung bang cu the.
FORCE_WIDE: set = set()


def load_force_wide(path: Path) -> None:
    global FORCE_WIDE
    if path.exists():
        FORCE_WIDE = {l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                      if l.strip() and not l.startswith("#")}


def widen_tables(body: str) -> str:
    """Bang viet cho elsarticle/acmart mot cot rong hon cot chu cua IEEEtran
    (~252pt), nen o ban IEEE chung tran sang cot ben canh.

    Quy tac cu la "tu 3 cot tro len -> table*". No qua manh tay: mot bang 3 cot
    voi o ngan van vua mot cot, va day no ra ca trang lam mat nua be rong cho
    khoang trang, doi lai la them nua trang giay. Quy tac moi doi hoi bang
    thuc su CAN be rong:

      - >= 5 cot, hoac o dai nhat > NEED_WIDE ky tu, hoac co cot p{} do dai
        tuyet doi  -> table*
      - 3-4 cot o ngan  -> giu mot cot, chi thu nho dem cot va co chu
      - 2 cot co o dai  -> giu mot cot nhung cho cot gia tri tu xuong dong
      - 2 cot o ngan    -> giu nguyen
    """
    out, i = [], 0
    pat = re.compile(r"\\begin\{table\}(\[[^\]]*\])?")
    while True:
        m = pat.search(body, i)
        if not m:
            out.append(body[i:])
            break
        end = body.find(r"\end{table}", m.end())
        if end < 0:
            out.append(body[i:])
            break
        out.append(body[i:m.start()])
        block = body[m.end():end]

        # IEEEtran KHONG tu can giua bang nhu acmart. Bang lay tu nguon acmart
        # khong co \centering nen o ban IEEE no dinh sat le trai, de lai mot
        # mang trang ben phai. Them \centering cho moi bang chua co.
        if "\\centering" not in block and "\\begin{center}" not in block:
            block = "\n\\centering" + block

        spec, span = _spec_of(block)
        if spec is None:
            spec, span = "", None
        ncol = _ncols(spec)
        longest = _longest_cell(block)

        def shrink(b):
            if not re.search(r"\\(footnotesize|small|scriptsize)\b", b):
                b = (b.replace(r"\centering", "\\centering\n\\footnotesize", 1)
                     if r"\centering" in b
                     else b.replace(r"\begin{tabular}",
                                    "\\footnotesize\n\\begin{tabular}", 1))
            if "tabcolsep" not in b:
                b = b.replace(r"\begin{tabular}",
                              "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}", 1)
            return b

        NEED_WIDE = 58   # do dai o (ky tu) tu do tro len thi bang can ca trang
        abs_p = bool(re.search(r"p\{\s*[\d.]+\s*(mm|cm|in|pt)\s*\}", spec))
        # Nhan cua bang, de doi chieu voi danh sach bang da do duoc la bi tran.
        _lb = re.search(r"\\label\{([^}]*)\}", block)
        forced = bool(_lb and _lb.group(1) in FORCE_WIDE)
        if ncol >= 3 and (forced or ncol >= 5 or longest > NEED_WIDE or abs_p):
            out.append(r"\begin{table*}[t]" + shrink(block) + r"\end{table*}")
        elif ncol >= 3:
            out.append(r"\begin{table}" + (m.group(1) or "[t]") + shrink(block)
                       + r"\end{table}")
        elif ncol == 2 and re.search(r"p\{\s*[\d.]+\s*(mm|cm|in|pt)\s*\}", spec):
            # p{82mm} duoc tinh cho cot rong cua elsarticle; o IEEEtran
            # (cot ~88mm) no day bang tran sang cot ben canh.
            ns = "@{}>{\\raggedright\\arraybackslash}p{0.32\\linewidth}@{\\hspace{5pt}}>{\\raggedright\\arraybackslash}p{0.62\\linewidth}@{}"
            b = block[:span[0]] + "\\begin{tabular}{" + ns + "}" + block[span[1]:]
            out.append(r"\begin{table}" + (m.group(1) or "[t]") + shrink(b)
                       + r"\end{table}")
        elif ncol == 2 and longest > 38 and "p{" not in spec:
            # cho cot cuoi tu xuong dong thay vi chay tran ra le
            newspec = "@{}>{\\raggedright\\arraybackslash}p{0.32\\linewidth}@{\\hspace{5pt}}>{\\raggedright\\arraybackslash}p{0.62\\linewidth}@{}"
            b = block[:span[0]] + "\\begin{tabular}{" + newspec + "}" \
                + block[span[1]:]
            b = b.replace(r"\\" + "\n" + r"\bottomrule",
                          r"\\[2pt]" + "\n" + r"\bottomrule", 1)
            out.append(r"\begin{table}" + (m.group(1) or "[t]") + shrink(b)
                       + r"\end{table}")
        else:
            out.append(r"\begin{table}" + (m.group(1) or "[t]") + block
                       + r"\end{table}")
        i = end + len(r"\end{table}")
    return "".join(out)


def convert(src: Path, dst: Path, keywords: str, lect):
    s = src.read_text(encoding="utf-8")

    # --- lay tieu de
    m = re.search(r"\\title(?:\[[^\]]*\])?\{(.*?)\}\s*\n", s, re.S)
    if not m:
        m = re.search(r"\\title\{(.*?)\}", s, re.S)
    title = m.group(1).strip() if m else "Untitled"
    title = re.sub(r"\s+", " ", title)

    # --- lay abstract
    ma = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    abstract = ma.group(1).strip() if ma else ""

    # --- lay khoi macro so lieu (\newcommand) trong phan dau file
    head = s[:s.find("\\section{")] if "\\section{" in s else s
    macros = "\n".join(m.group(0) for m in
                        re.finditer(r"(?m)^\\newcommand\{\\[A-Za-z]+\}\{.*\}$", head))

    # --- lay than bai: tu \section dau tien den het
    i = s.find("\\section{")
    body = s[i:] if i > 0 else s
    # bo phan bibliography cu neu co
    body = re.sub(r"\\bibliographystyle\{[^}]*\}\s*", "", body)
    body = re.sub(r"\\bibliography\{[^}]*\}\s*", "", body)
    body = re.sub(r"\\end\{document\}\s*$", "", body)
    # acmart: bo cac lenh rieng. Giu lai loi khai bao tai tro trong \begin{acks}
    # de chuyen thanh \thanks o trang 1 (dung le IEEE).
    funding = ""
    ma_ = re.search(r"\\begin\{acks\}(.*?)\\end\{acks\}", body, re.S)
    if ma_:
        txt = " ".join(ma_.group(1).split())
        for sent in re.split(r"(?<=\.)\s+", txt):
            if re.search(r"(?i)funding|grant|supported by", sent):
                funding = sent.strip()
                break
    body = re.sub(r"\\begin\{acks\}.*?\\end\{acks\}", "", body, flags=re.S)
    body = re.sub(r"\\Description\{.*?\}", "", body, flags=re.S)

    # CRediT la le cua Elsevier/ACM. IEEE khong co truong nay trong ban thao,
    # nen bo khoi ban IEEE (van giu nguyen trong ban goc elsarticle/acmart).
    body = re.sub(r"\\section\*\{CRediT [Aa]uthorship [Cc]ontribution [Ss]tatement\}"
                  r".*?(?=\n\\section\*\{|\n\\bibliographystyle|\Z)",
                  "", body, flags=re.S)

    # IEEE dat loi khai bao tai tro o chu thich trang 1, khong lam mot muc rieng.
    mf = re.search(r"\\section\*\{Funding\}\s*\n(.*?)(?=\n\\section\*\{|\n\\bibliographystyle|\Z)",
                   body, re.S)
    if mf:
        funding = " ".join(mf.group(1).split())
        body = body[:mf.start()] + body[mf.end():]

    body = widen_tables(body)
    body = center_wide_and_figs(body)
    body = break_long_tt(body)
    body = widen_overview_figure(body)

    out = (
        "%% ------------------------------------------------------------------\n"
        "%% Ban IEEE: IEEEtran, journal mode, 2 cot, 10pt\n"
        "%% Sinh tu paper_template.tex boi ieee/to_ieee.py -- dung sua tay\n"
        "%% ------------------------------------------------------------------\n"
        "\\documentclass[journal]{IEEEtran}\n"
        + PREAMBLE_EXTRA
        + "\n%% --- macro so lieu, sinh tu make_tables.py ---\n"
        + macros + "\n"
        + "\n\\begin{document}\n\n"
        + "\\title{" + title + "}\n\n"
        + authorblock(lect)
        + "\n\\maketitle\n\n"
        + "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n"
        + "\\begin{IEEEkeywords}\n" + keywords + "\n\\end{IEEEkeywords}\n\n"
        + body
        + "\n\\bibliographystyle{IEEEtran}\n\\bibliography{refs}\n\n"
        + "\\end{document}\n"
    )
    if funding:
        anchor = "  \\thanks{\\emph{(Corresponding author:"
        assert anchor in out, "khoi tac gia thieu dong corresponding author"
        out = out.replace(anchor, "  \\thanks{" + funding + "}%\n" + anchor, 1)

    dst.write_text(out, encoding="utf-8")
    return len(abstract.split())


if __name__ == "__main__":
    src, dst, kw = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    # So bai lay tu duong dan, vi du /home/claude/p07/latex/main.tex -> "07"
    m = re.search(r"/p(\d\d)/", str(src.resolve()))
    lect = sys.argv[4] if len(sys.argv) > 4 else ASSIGN[m.group(1)]
    load_force_wide(dst.parent / "force_wide.txt")
    n = convert(src, dst, kw, lect)
    print(f"  {dst} viet xong (abstract ~{n} tu)")
