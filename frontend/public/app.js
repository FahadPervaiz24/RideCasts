const { Deck, GeoJsonLayer } = deck;

const ZONES_URL = "./data/taxi_zones.geojson";
const FORECAST_URLS = [
  "./data/forecast_latest.json",
  "/data/forecast/forecast_latest.json",
];

const state = {
  zones: null,
  forecast: null,
  hours: [],
  selectedHourIndex: 0,
  lookup: new Map(),
  zoneNameById: new Map(),
  currentRows: [],
  deck: null,
  maxPred: 1,
  playTimer: null,
};

const VIEW_LIMITS = {
  minLon: -74.8,
  maxLon: -73.2,
  minLat: 40.2,
  maxLat: 41.1,
  minZoom: 8.2,
  maxZoom: 13.8,
};

function isMobileViewport() {
  return window.matchMedia("(max-width: 900px)").matches;
}

const el = {
  hourSlider: document.getElementById("hourSlider"),
  playBtn: document.getElementById("playBtn"),
  hourLabel: document.getElementById("hourLabel"),
  topZones: document.getElementById("topZones"),
  tooltip: document.getElementById("mapTooltip"),
  loadingOverlay: document.getElementById("loadingOverlay"),
};

function syncVisualViewportVars() {
  const vv = window.visualViewport;
  if (!vv) return;
  document.documentElement.style.setProperty("--vvh", `${Math.round(vv.height)}px`);
  document.documentElement.style.setProperty("--vv-top", `${Math.round(vv.offsetTop)}px`);
}

function hideLoading() {
  if (!el.loadingOverlay) return;
  el.loadingOverlay.classList.add("hidden");
}

function showLoadingError(message) {
  if (!el.loadingOverlay) return;
  el.loadingOverlay.classList.remove("hidden");
  el.loadingOverlay.classList.add("error");
  el.loadingOverlay.innerHTML = `
    <div class="loading-card">
      <div class="loading-title">Unable to load forecast</div>
      <div class="loading-sub">${message}</div>
    </div>
  `;
}

function clamp(val, min, max) {
  return Math.min(max, Math.max(min, val));
}

function clampViewState(viewState) {
  return {
    ...viewState,
    longitude: clamp(viewState.longitude, VIEW_LIMITS.minLon, VIEW_LIMITS.maxLon),
    latitude: clamp(viewState.latitude, VIEW_LIMITS.minLat, VIEW_LIMITS.maxLat),
    zoom: clamp(viewState.zoom, VIEW_LIMITS.minZoom, VIEW_LIMITS.maxZoom),
  };
}

async function fetchFirstAvailable(urls) {
  for (const url of urls) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (res.ok) return res.json();
    } catch (_) {
      // try next path
    }
  }
  throw new Error("Could not load forecast file from known paths.");
}

function predictionColor(value) {
  const v = Math.max(0, value || 0);
  if (v < 25) return [67, 150, 185, 210]; // #4396B9
  if (v < 75) return [120, 224, 205, 210]; // #78E0CD
  if (v < 150) return [224, 253, 188, 215]; // #E0FDBC
  if (v < 350) return [251, 237, 184, 220]; // #FBEDB8
  if (v < 750) return [242, 177, 99, 225]; // #F2B163
  return [193, 68, 82, 235]; // #C14452
}

function formatHourLabel(isoString) {
  const d = new Date(isoString);
  return d.toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function setList(root, rows) {
  root.innerHTML = "";
  rows.forEach((r) => {
    const li = document.createElement("li");
    const zoneName = state.zoneNameById.get(Number(r.PULocationID)) || `Zone ${r.PULocationID}`;
    const trips = Math.round(r.prediction).toLocaleString("en-US");
    li.textContent = `${zoneName}: ${trips} trips`;
    root.appendChild(li);
  });
}

function updateStats(hour) {
  const rows = state.currentRows;
  const sorted = [...rows].sort((a, b) => b.prediction - a.prediction);
  el.hourLabel.textContent = formatHourLabel(hour);
  setList(el.topZones, sorted.slice(0, 5));
}

function renderMap() {
  const layer = new GeoJsonLayer({
    id: "zones-layer",
    data: state.zones,
    pickable: true,
    stroked: true,
    filled: true,
    extruded: true,
    wireframe: false,
    lineWidthMinPixels: 1.8,
    getLineColor: [222, 236, 250, 210],
    getFillColor: (f) => {
      const zone = Number(f.properties.PULocationID);
      const pred = state.lookup.get(zone) || 0;
      return predictionColor(pred);
    },
    getElevation: (f) => {
      const zone = Number(f.properties.PULocationID);
      const pred = state.lookup.get(zone) || 0;
      return Math.sqrt(pred) * 35;
    },
    updateTriggers: {
      getFillColor: [state.selectedHourIndex],
      getElevation: [state.selectedHourIndex],
    },
    transitions: {
      getFillColor: 450,
      getElevation: 450,
    },
    onHover: ({ object, x, y }) => {
      if (!object) {
        el.tooltip.style.display = "none";
        return;
      }

      const zone = Number(object.properties.PULocationID);
      const pred = state.lookup.get(zone) || 0;
      const zoneName = object.properties.zone || `Zone ${zone}`;
      el.tooltip.innerHTML = `
        <div class="tooltip-title">${zoneName}</div>
        <div class="tooltip-row">TLC Zone: ${zone}</div>
        <div class="tooltip-row">Predicted trips: ${Math.round(pred)}</div>
      `;

      el.tooltip.style.display = "block";
      const mapRect = document.getElementById("map").getBoundingClientRect();
      const tipRect = el.tooltip.getBoundingClientRect();
      const pad = 12;
      let left = x + pad;
      let top = y + pad;
      left = Math.min(left, mapRect.width - tipRect.width - pad);
      top = Math.min(top, mapRect.height - tipRect.height - pad);
      left = Math.max(pad, left);
      top = Math.max(pad, top);
      el.tooltip.style.transform = `translate(${left}px, ${top}px)`;
    },
  });

  state.deck.setProps({ layers: [layer] });
}

function updateHour(index) {
  state.selectedHourIndex = Number(index);
  const hour = state.hours[state.selectedHourIndex];
  const rows = state.forecast.predictions.filter((p) => p.hour === hour);
  state.currentRows = rows;
  state.lookup = new Map(rows.map((r) => [Number(r.PULocationID), Number(r.prediction)]));
  state.maxPred = Math.max(1, ...rows.map((r) => Number(r.prediction)));
  el.hourSlider.value = String(state.selectedHourIndex);
  updateStats(hour);
  renderMap();
}

function stopPlayback() {
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
  el.playBtn.textContent = "▶";
  el.playBtn.setAttribute("aria-label", "Play");
}

function startPlayback() {
  stopPlayback();
  el.playBtn.textContent = "II";
  el.playBtn.setAttribute("aria-label", "Pause");
  state.playTimer = setInterval(() => {
    const max = Math.max(0, state.hours.length - 1);
    let next = state.selectedHourIndex + 1;
    if (next > max) next = 0;
    updateHour(next);
  }, 420);
}

async function init() {
  syncVisualViewportVars();
  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", syncVisualViewportVars);
    window.visualViewport.addEventListener("scroll", syncVisualViewportVars);
  }

  const [zonesRes, forecast] = await Promise.all([
    fetch(ZONES_URL).then((r) => r.json()),
    fetchFirstAvailable(FORECAST_URLS),
  ]);

  state.zones = zonesRes;
  state.forecast = forecast;
  state.hours = [...new Set(forecast.predictions.map((p) => p.hour))].sort();
  state.zoneNameById = new Map(
    state.zones.features.map((f) => [
      Number(f.properties.PULocationID),
      f.properties.zone || `Zone ${f.properties.PULocationID}`,
    ])
  );

  el.hourSlider.min = "0";
  el.hourSlider.max = String(Math.max(0, state.hours.length - 1));
  el.hourSlider.value = "0";
  el.hourSlider.step = "1";
  el.hourSlider.addEventListener("input", (e) => {
    stopPlayback();
    updateHour(e.target.value);
  });

  el.playBtn.addEventListener("click", () => {
    if (state.playTimer) stopPlayback();
    else startPlayback();
  });

  state.deck = new Deck({
    parent: document.getElementById("map"),
    initialViewState: isMobileViewport()
      ? {
          longitude: -73.95,
          latitude: 40.73,
          zoom: 9.6,
          pitch: 35,
          bearing: 0,
          minZoom: VIEW_LIMITS.minZoom,
          maxZoom: VIEW_LIMITS.maxZoom,
        }
      : {
          longitude: -73.95,
          latitude: 40.73,
          zoom: 10,
          pitch: 45,
          bearing: 0,
          minZoom: VIEW_LIMITS.minZoom,
          maxZoom: VIEW_LIMITS.maxZoom,
        },
    controller: {
      minZoom: VIEW_LIMITS.minZoom,
      maxZoom: VIEW_LIMITS.maxZoom,
      minPitch: 0,
      maxPitch: 60,
    },
    onViewStateChange: ({ viewState }) => clampViewState(viewState),
    layers: [],
  });

  updateHour(0);
  hideLoading();

  window.addEventListener("resize", () => {
    const mobile = isMobileViewport();
    const next = mobile
      ? { longitude: -73.95, latitude: 40.73, zoom: 9.6, pitch: 35, bearing: 0 }
      : { longitude: -73.95, latitude: 40.73, zoom: 10, pitch: 45, bearing: 0 };
    state.deck.setProps({ viewState: clampViewState(next) });
  });
}

init().catch((err) => {
  console.error(err);
  showLoadingError("Please refresh in a moment.");
});
