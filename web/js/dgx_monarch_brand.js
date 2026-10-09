// Shared artwork for the Monarch panel and its sidebar button.
import { el } from "./dgx_monarch_ui.js";

const assets = {
    full: new URL("../brand/monarch.svg", import.meta.url).href,
    fullDark: new URL("../brand/monarch-dark.svg", import.meta.url).href,
    small: new URL("../brand/monarch-small.svg", import.meta.url).href,
    smallDark: new URL("../brand/monarch-small-dark.svg", import.meta.url).href,
};

export function updateBrand(mark, context) {
    // ComfyUI themes supply the foreground color, including custom palettes.
    const color = getComputedStyle(context).color;
    const rgb = color.match(/^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/);
    const bright = rgb ? (Number(rgb[1]) * .2126 + Number(rgb[2]) * .7152 +
        Number(rgb[3]) * .0722) > 128 : true;
    const size = mark.dataset.dgxmSize;
    const image = mark.querySelector("img");
    const source = assets[size + (bright ? "Dark" : "")];
    if (image.src !== source) image.src = source;
}

export function brandMark(context, small = false) {
    const size = small ? 28 : 40;
    const mark = el("span", `display:inline-flex;align-items:center;justify-content:center;` +
        `width:${size}px;height:${size}px;flex:0 0 ${size}px;`);
    mark.dataset.dgxmSize = small ? "small" : "full";
    mark.setAttribute("aria-hidden", "true");
    const image = el("img", "display:block;width:100%;height:100%;object-fit:contain;");
    image.alt = "";
    image.width = image.height = size;
    image.draggable = false;
    const fallback = el("span", "font-size:10px;font-weight:700;letter-spacing:.3px;", "DGX");
    fallback.hidden = true;
    image.onerror = () => { image.hidden = true; image.style.display = "none"; fallback.hidden = false; };
    image.onload = () => { image.hidden = false; image.style.display = "block"; fallback.hidden = true; };
    mark.appendChild(image);
    mark.appendChild(fallback);
    updateBrand(mark, context);
    return mark;
}

export function brandHeader(context) {
    const header = el("div", "display:flex;align-items:center;gap:10px;padding:2px 0 12px;");
    header.appendChild(brandMark(context));
    const label = el("div", "min-width:0;");
    label.appendChild(el("div", "font-size:16px;font-weight:650;letter-spacing:.2px;", "DGX Monarch"));
    label.appendChild(el("div", "font-size:11px;opacity:.65;margin-top:2px;", "Cluster status"));
    header.appendChild(label);
    return header;
}
