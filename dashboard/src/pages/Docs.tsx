import { motion } from "framer-motion";
import { useEffect, useState, type ReactNode } from "react";
import { api } from "../lib/api";

function inline(s: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0, m: RegExpExecArray | null, i = 0;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(s.slice(last, m.index));
    const t = m[0];
    if (t.startsWith("**")) out.push(<b key={key + i++}>{t.slice(2, -2)}</b>);
    else if (t.startsWith("`")) out.push(<code key={key + i++}>{t.slice(1, -1)}</code>);
    else {
      const [, label, href] = /\[([^\]]+)\]\(([^)]+)\)/.exec(t)!;
      out.push(href.startsWith("http") ? <a key={key + i++} href={href} target="_blank" rel="noreferrer">{label}</a> : <span key={key + i++}>{label}</span>);
    }
    last = m.index + t.length;
  }
  if (last < s.length) out.push(s.slice(last));
  return out;
}

export function Markdown({ text }: { text: string }) {
  const lines = text.split(/\r?\n/);
  const out: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    const k = `l${i}`;
    if (l.startsWith("```")) {
      const buf: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) buf.push(lines[i++]);
      i++;
      out.push(<pre key={k}>{buf.join("\n")}</pre>);
      continue;
    }
    if (l.startsWith("|")) {
      const rows: string[] = [];
      while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
      const cells = (r: string) => r.split("|").slice(1, -1).map((c) => c.trim());
      out.push(
        <table key={k}>
          <thead><tr>{cells(rows[0]).map((c, j) => <th key={j}>{inline(c, k + j)}</th>)}</tr></thead>
          <tbody>{rows.slice(2).map((r, ri) => <tr key={ri}>{cells(r).map((c, j) => <td key={j}>{inline(c, k + ri + j)}</td>)}</tr>)}</tbody>
        </table>,
      );
      continue;
    }
    if (l.startsWith(">")) {
      const buf: string[] = [];
      while (i < lines.length && lines[i].startsWith(">")) buf.push(lines[i++].replace(/^>\s?/, ""));
      out.push(<blockquote key={k}>{inline(buf.join(" "), k)}</blockquote>);
      continue;
    }
    if (/^\s*[-*] /.test(l)) {
      const buf: string[] = [];
      while (i < lines.length && /^\s*[-*] /.test(lines[i])) buf.push(lines[i++].replace(/^\s*[-*] /, ""));
      out.push(<ul key={k}>{buf.map((b, j) => <li key={j}>{inline(b, k + j)}</li>)}</ul>);
      continue;
    }
    if (/^\d+\. /.test(l)) {
      const buf: string[] = [];
      while (i < lines.length && /^\d+\. /.test(lines[i])) buf.push(lines[i++].replace(/^\d+\. /, ""));
      out.push(<ol key={k}>{buf.map((b, j) => <li key={j}>{inline(b, k + j)}</li>)}</ol>);
      continue;
    }
    if (l.startsWith("### ")) out.push(<h3 key={k}>{inline(l.slice(4), k)}</h3>);
    else if (l.startsWith("## ")) out.push(<h2 key={k}>{inline(l.slice(3), k)}</h2>);
    else if (l.startsWith("# ")) out.push(<h1 key={k}>{inline(l.slice(2), k)}</h1>);
    else if (l.trim()) out.push(<p key={k}>{inline(l, k)}</p>);
    i++;
  }
  return <div className="md">{out}</div>;
}

export function DocsPage() {
  const [docs, setDocs] = useState<{ name: string; title: string }[]>([]);
  const [cur, setCur] = useState<string | null>(null);
  const [text, setText] = useState("");
  useEffect(() => {
    api.get<{ name: string; title: string }[]>("/api/docs").then((d) => {
      setDocs(d);
      setCur(d.find((x) => x.name === "PROJECT_OVERVIEW.md")?.name ?? d[0]?.name ?? null);
    });
  }, []);
  useEffect(() => {
    if (cur) api.get<string>(`/api/docs/${encodeURIComponent(cur)}`).then(setText);
  }, [cur]);
  return (
    <>
      <div className="page-head">
        <div>
          <h2>Documentation</h2>
          <p>What was built, how each piece maps to the paper, how to reproduce it, and the generated evaluation reports.</p>
        </div>
      </div>
      <div className="grid" style={{ gridTemplateColumns: "minmax(180px, 240px) minmax(0, 1fr)" }}>
        <div className="panel" style={{ alignSelf: "start", display: "flex", flexDirection: "column", gap: 4 }}>
          {docs.map((d) => (
            <button key={d.name} className={`navbtn ${cur === d.name ? "active" : ""}`} onClick={() => setCur(d.name)}>
              {cur === d.name && <motion.span layoutId="doc-bg" className="pill-bg" />}
              <span style={{ fontSize: 13 }}>{d.title || d.name}</span>
            </button>
          ))}
        </div>
        <motion.div key={cur} className="panel" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <Markdown text={text} />
        </motion.div>
      </div>
    </>
  );
}
