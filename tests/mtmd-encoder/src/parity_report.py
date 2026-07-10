"""parity_results.md renderer for the mtmd-encoder harnesses; see README.md."""


def write_parity_results(md_path, quants, cmds, results):
    fam = md_path.parent.name
    lines = [f"# {fam} vision parity results", ""]
    for quant in quants:
        metrics = results[quant]
        lines += [f"## {quant}", "", f"`{' '.join(cmds[quant])}`", ""]
        lines += ["| image | min_cosim | worst_rel_l2 |", "|---|---:|---:|"]
        for image in sorted(metrics):
            m = metrics[image]
            if "per_token_cosim_min" in m:
                lines.append(f"| {image} | {m['per_token_cosim_min']:.10f} | {m['per_token_l2_worst']:.10f} |")
            else:
                lines.append(f"| {image} | - | - |")
        lines.append("")

    for title, key, agg in [
        ("worst min_cosim", "per_token_cosim_min", min),
        ("worst worst_rel_l2", "per_token_l2_worst", max),
    ]:
        lines += [f"## Summary - {title}", ""]
        lines += ["| quant | value | view |", "|---|---:|---|"]
        for quant in quants:
            vals = [(m[key], image) for image, m in results[quant].items() if key in m]
            if vals:
                val, image = agg(vals)
                lines.append(f"| {quant} | {val:.10f} | {image} |")
            else:
                lines.append(f"| {quant} | - | - |")
        lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"wrote {md_path}")
