/**
 * Crypto Forecasting Dashboard
 * 14-day forecast path, historical + future chart, auto-refresh.
 */

(function () {
  'use strict';

  const API_BASE = '';
  const coinSelect = document.getElementById('coin-select');
  const fetchBtn = document.getElementById('fetch-btn');
  const loadingEl = document.getElementById('loading');
  const predictionEl = document.getElementById('prediction');
  const chartSection = document.getElementById('chart-section');
  const forecastTableSection = document.getElementById('forecast-table-section');
  const evaluationSection = document.getElementById('evaluation-section');
  const errorEl = document.getElementById('error');
  const autoRefreshToggle = document.getElementById('auto-refresh-toggle');
  const refreshIntervalInput = document.getElementById('refresh-interval');

  let refreshTimer = null;
  const DEFAULT_REFRESH_SEC = 60;

  function hideAll() {
    loadingEl.classList.add('hidden');
    predictionEl.classList.add('hidden');
    chartSection.classList.add('hidden');
    forecastTableSection.classList.add('hidden');
    evaluationSection.classList.add('hidden');
    errorEl.classList.add('hidden');
  }

  function showError(msg) {
    hideAll();
    errorEl.textContent = msg;
    errorEl.classList.remove('hidden');
  }

  /** Adaptive formatting for large vs micro-price coins (Bitcoin vs PEPE / SHIB) */
  function formatPrice(n, referencePrice) {
    if (typeof n !== 'number' || isNaN(n)) return '—';
    const ref = referencePrice != null && referencePrice > 0 ? referencePrice : Math.abs(n) || 1;
    if (ref >= 1e6) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (ref >= 1) return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (ref >= 0.01) return n.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 6 });
    var logr = Math.log10(Math.max(ref, 1e-30));
    var frac = Math.min(14, Math.max(8, 4 - Math.floor(logr)));
    return n.toFixed(frac);
  }

  function formatNumber(n) {
    if (typeof n !== 'number' || isNaN(n)) return '—';
    if (Math.abs(n) >= 1e6) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (Math.abs(n) >= 1) return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return n.toFixed(4);
  }

  function formatDate(iso) {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      return d.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' });
    } catch (_) {
      return iso;
    }
  }

  function formatDateOnly(iso) {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    } catch (_) {
      return String(iso).slice(0, 10);
    }
  }

  function formatPercent(n) {
    if (typeof n !== 'number' || isNaN(n)) return '—';
    return n.toFixed(2) + '%';
  }

  function trendDisplay(trendLabel) {
    var display = (trendLabel != null && String(trendLabel).trim() !== '') ? trendLabel : 'NEUTRAL';
    var label = String(display).trim().toUpperCase();
    var arrows = {
      'STRONG UP': '↑↑',
      'UP': '↑',
      'NEUTRAL': '→',
      'SIDEWAYS': '→',
      'DOWN': '↓',
      'STRONG DOWN': '↓↓',
    };
    var arrow = arrows[label] || '→';
    var cls = 'trend-sideways';
    if (label === 'STRONG UP') cls = 'trend-strong-up';
    else if (label === 'UP') cls = 'trend-up';
    else if (label === 'NEUTRAL' || label === 'SIDEWAYS') cls = 'trend-sideways';
    else if (label === 'DOWN') cls = 'trend-down';
    else if (label === 'STRONG DOWN') cls = 'trend-strong-down';
    return { text: arrow + ' ' + display, className: cls };
  }

  function setConfidenceUI(score) {
    var pctEl = document.getElementById('pred-confidence-pct');
    var fillEl = document.getElementById('pred-confidence-fill');
    if (!pctEl || !fillEl) return;
    if (typeof score !== 'number' || isNaN(score)) {
      pctEl.textContent = '—';
      fillEl.style.width = '0%';
      fillEl.className = 'confidence-bar-fill';
      return;
    }
    var p = Math.max(0, Math.min(1, score)) * 100;
    pctEl.textContent = p.toFixed(1) + '%';
    fillEl.style.width = p + '%';
    fillEl.className = 'confidence-bar-fill ' + (p >= 66 ? 'conf-high' : p >= 40 ? 'conf-mid' : 'conf-low');
  }

  /** Plotly y-axis tickformat for adaptive scaling (micro-prices) */
  function chartTickFormat(refPrice) {
    if (refPrice == null || refPrice <= 0) return '$.4f';
    if (refPrice < 0.0001) return '.2e';
    if (refPrice < 1) return '.6f';
    return '$,.2f';
  }

  async function loadCoins() {
    try {
      const res = await fetch(API_BASE + '/coins');
      if (!res.ok) throw new Error('Failed to load coins');
      const data = await res.json();
      coinSelect.innerHTML = '<option value="">Select coin…</option>';
      (data.coins || []).forEach(function (c) {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        coinSelect.appendChild(opt);
      });
    } catch (err) {
      showError('Could not load coin list: ' + (err.message || err));
    }
  }

  function renderPrediction(data) {
    const refPrice = data.current_price || data.average_prediction;
    document.getElementById('pred-coin').textContent = data.coin || '—';
    document.getElementById('pred-current').textContent = formatPrice(data.current_price, refPrice);
    document.getElementById('pred-avg').textContent = formatPrice(
      data.summary ? data.summary.final_day_prediction : data.average_prediction,
      refPrice
    );
    document.getElementById('pred-bounds').textContent =
      formatPrice(data.summary ? data.summary.min_forecast_price : data.lower_bound, refPrice) +
      ' / ' +
      formatPrice(data.summary ? data.summary.max_forecast_price : data.upper_bound, refPrice);
    document.getElementById('pred-agreement').textContent =
      typeof data.model_agreement_score === 'number'
        ? formatPercent(data.model_agreement_score * 100)
        : data.forecast_path && data.forecast_path.length
          ? formatPercent((data.forecast_path[data.forecast_path.length - 1].agreement_score || 0) * 100)
          : '—';
    document.getElementById('pred-horizon').textContent =
      (data.horizon_days != null ? data.horizon_days : 14) + ' days';
    var periodStart = data.forecast_period_start;
    var periodEnd = data.forecast_period_end;
    document.getElementById('pred-forecast-period').textContent =
      (periodStart && periodEnd)
        ? formatDateOnly(periodStart) + ' – ' + formatDateOnly(periodEnd)
        : '—';
    document.getElementById('pred-market-ts').textContent =
      formatDate(data.latest_market_timestamp) || '—';
    document.getElementById('pred-generated').textContent = formatDate(data.generated_at);
    var freshnessEl = document.getElementById('pred-freshness');
    var freshnessMsgEl = document.getElementById('pred-freshness-msg');
    if (freshnessEl) {
      var freshness = (data.data_freshness || 'unknown').toLowerCase();
      var ageH = data.data_age_hours;
      var label = freshness === 'fresh' || freshness === 'stale' ? freshness : '—';
      if ((freshness === 'fresh' || freshness === 'stale') && typeof ageH === 'number') {
        label += ' (' + ageH + 'h)';
      }
      freshnessEl.textContent = label;
      freshnessEl.className = 'value freshness-value' + (freshness === 'fresh' ? ' freshness-fresh' : freshness === 'stale' ? ' freshness-stale' : '');
    }
    if (freshnessMsgEl) {
      var msg = data.data_freshness_message || data.data_freshness_detail || '';
      freshnessMsgEl.textContent = msg;
      freshnessMsgEl.classList.toggle('hidden', !msg);
    }

    setConfidenceUI(data.confidence_score);

    var trendEl = document.getElementById('pred-trend');
    if (trendEl) {
      var tl = data.trend_label || (data.summary && data.summary.trend_label);
      var td = trendDisplay(tl || '');
      trendEl.textContent = td.text;
      trendEl.className = 'value trend-value ' + td.className;
    }

    var volEl = document.getElementById('pred-volatility');
    if (volEl) {
      volEl.textContent = data.volatility_level || '—';
    }

    var meanAgEl = document.getElementById('pred-mean-agreement');
    if (meanAgEl) {
      meanAgEl.textContent =
        typeof data.mean_path_agreement === 'number'
          ? formatPercent(data.mean_path_agreement * 100)
          : '—';
    }

    var tsBlock = document.getElementById('trade-signal-block');
    var tdBadge = document.getElementById('trade-decision-badge');
    var teScore = document.getElementById('trade-edge-score');
    var trList = document.getElementById('trade-reasons-list');
    if (tsBlock && tdBadge && teScore && trList) {
      var dec = data.trade_decision != null ? String(data.trade_decision) : 'NO_TRADE';
      tdBadge.textContent = dec.replace(/_/g, ' ');
      tdBadge.className =
        'trade-decision-badge ' +
        (dec === 'STRONG_BUY' ? 'td-strong' : dec === 'WEAK_BUY' ? 'td-weak' : 'td-none');
      if (typeof data.edge_score === 'number' && !isNaN(data.edge_score)) {
        var es = Math.max(0, Math.min(1, data.edge_score));
        teScore.textContent = es.toFixed(3) + ' (' + (es * 100).toFixed(1) + '% scale)';
      } else {
        teScore.textContent = '—';
      }
      trList.innerHTML = '';
      var reasons = data.trade_reasons;
      if (reasons && reasons.length) {
        reasons.forEach(function (r) {
          var li = document.createElement('li');
          li.textContent = r;
          trList.appendChild(li);
        });
      } else {
        var li0 = document.createElement('li');
        li0.className = 'trade-reason-muted';
        li0.textContent = 'No supplemental reasons.';
        trList.appendChild(li0);
      }
      tsBlock.classList.remove('hidden');
    }

    function riskMeterLabel(score) {
      if (typeof score !== 'number' || isNaN(score)) return '—';
      var s = Math.max(0, Math.min(1, score));
      var pct = (s * 100).toFixed(0);
      var tier = s >= 0.65 ? 'strong' : s >= 0.45 ? 'mid' : 'weak';
      return { text: pct + '%', tier: tier };
    }

    var riBlock = document.getElementById('risk-intel-block');
    var rs = document.getElementById('risk-stability');
    var rc = document.getElementById('risk-consensus');
    var rt = document.getElementById('risk-trend-conf');
    var rk = document.getElementById('risk-shock');
    if (riBlock && rs && rc && rt && rk) {
      var ms = riskMeterLabel(data.stability_score);
      var mc = riskMeterLabel(data.consensus_score);
      var mt = riskMeterLabel(data.trend_confirmation_score);
      rs.textContent = ms.text;
      rs.className = 'value risk-meter risk-' + ms.tier;
      rc.textContent = mc.text;
      rc.className = 'value risk-meter risk-' + mc.tier;
      rt.textContent = mt.text;
      rt.className = 'value risk-meter risk-' + mt.tier;
      if (data.volatility_shock_detected === true) {
        rk.textContent = 'Yes — elevated';
        rk.className = 'value risk-shock-yes';
      } else {
        rk.textContent = 'No';
        rk.className = 'value risk-shock-no';
      }
      riBlock.classList.remove('hidden');
    }

    var qStrip = document.getElementById('pred-quality-strip');
    if (qStrip) {
      var am = data.artifact_mode || '—';
      var fq = data.forecast_quality || '—';
      var sm = data.schema_match === true ? 'schema OK' : data.schema_match === false ? 'schema mismatch' : '—';
      var fv = data.forecast_validity || '—';
      var fqs =
        typeof data.forecast_quality_score === 'number' && !isNaN(data.forecast_quality_score)
          ? (data.forecast_quality_score * 100).toFixed(1) + '%'
          : '—';
      var warn =
        data.is_constant_prediction || data.low_variance_warning || data.degraded_input || data.fallback_mode;
      qStrip.className = 'quality-strip' + (warn ? ' quality-warn' : '');
      qStrip.innerHTML =
        '<strong>Quality</strong> ' +
        fq +
        ' · <strong>Artifact</strong> ' +
        am +
        ' · ' +
        sm +
        ' · <strong>Validity</strong> ' +
        fv +
        ' · <strong>Path score</strong> ' +
        fqs;
      qStrip.classList.remove('hidden');
    }

    var explBlock = document.getElementById('pred-explanation-block');
    var explEl = document.getElementById('pred-explanation');
    if (explBlock && explEl) {
      var ex = (data.explanation || '').trim();
      explEl.textContent = ex;
      explBlock.classList.toggle('hidden', !ex);
    }

    var mhBlock = document.getElementById('pred-multi-horizon-block');
    var mhEl = document.getElementById('pred-multi-horizon');
    if (mhBlock && mhEl) {
      mhEl.innerHTML = '';
      var mh = data.multi_horizon;
      var keys = mh && typeof mh === 'object' ? Object.keys(mh) : [];
      if (keys.length) {
        keys.forEach(function (k) {
          var s = mh[k];
          if (!s) return;
          var chip = document.createElement('span');
          chip.className = 'horizon-chip';
          var price = s.predicted_price;
          var ret = s.implied_return_vs_spot;
          var retStr =
            typeof ret === 'number'
              ? ' (' + (ret >= 0 ? '+' : '') + (ret * 100).toFixed(2) + '% vs spot)'
              : '';
          chip.innerHTML =
            '<strong>' +
            k +
            '</strong> ' +
            formatPrice(price, refPrice) +
            retStr;
          mhEl.appendChild(chip);
        });
        mhBlock.classList.remove('hidden');
      } else {
        mhBlock.classList.add('hidden');
      }
    }

    var wUl = document.getElementById('pred-weights');
    if (wUl) {
      wUl.innerHTML = '';
      var mw = data.model_weights && typeof data.model_weights === 'object' ? data.model_weights : {};
      var wkeys = Object.keys(mw);
      var wmax = 0;
      wkeys.forEach(function (k) {
        if (mw[k] > wmax) wmax = mw[k];
      });
      wkeys
        .sort(function (a, b) {
          return (mw[b] || 0) - (mw[a] || 0);
        })
        .forEach(function (model) {
          var w = mw[model];
          var li = document.createElement('li');
          var pct = wmax > 0 ? Math.round((w / wmax) * 100) : 0;
          li.innerHTML =
            '<span>' +
            model.replace(/_/g, ' ') +
            '</span><span class="w-bar-wrap"><span class="w-bar" style="width:' +
            pct +
            '%"></span></span><span>' +
            (typeof w === 'number' ? (w * 100).toFixed(1) + '%' : '—') +
            '</span>';
          wUl.appendChild(li);
        });
    }

    const ul = document.getElementById('pred-models');
    ul.innerHTML = '';
    const preds = data.model_predictions || (data.forecast_path && data.forecast_path[13]
      ? data.forecast_path[13].model_predictions : {});
    Object.keys(preds).forEach(function (model) {
      const li = document.createElement('li');
      li.innerHTML = '<span>' + model + '</span><span>' + formatPrice(preds[model], refPrice) + '</span>';
      ul.appendChild(li);
    });
  }

  function renderForecastTable(forecastPath, refPrice) {
    const tbody = document.getElementById('forecast-tbody');
    tbody.innerHTML = '';
    if (!forecastPath || forecastPath.length === 0) return;
    forecastPath.forEach(function (row) {
      const tr = document.createElement('tr');
      tr.innerHTML =
        '<td class="num">' + row.day_index + '</td>' +
        '<td>' + formatDateOnly(row.forecast_date || row.forecast_timestamp) + '</td>' +
        '<td class="num">' + formatPrice(row.predicted_price || row.ensemble_prediction, refPrice) + '</td>' +
        '<td class="num">' + formatPrice(row.lower_bound, refPrice) + '</td>' +
        '<td class="num">' + formatPrice(row.upper_bound, refPrice) + '</td>' +
        '<td class="num">' + formatPercent((row.agreement_score || 0) * 100) + '</td>';
      tbody.appendChild(tr);
    });
  }

  function renderChart(chartData, forecastPath, refPrice, latestMarketDate) {
    const container = document.getElementById('chart');
    container.innerHTML = '';

    const traces = [];
    const allDates = [];
    const allY = [];
    const isMicroPrice = refPrice != null && refPrice > 0 && refPrice < 0.01;

    if (chartData && chartData.dates && chartData.close && chartData.dates.length > 0) {
      traces.push({
        x: chartData.dates,
        y: chartData.close,
        type: 'scatter',
        mode: 'lines',
        name: 'Historical',
        line: { color: '#58a6ff', width: 2 },
        fill: 'tozeroy',
        fillcolor: 'rgba(88, 166, 255, 0.08)',
      });
      allDates.push.apply(allDates, chartData.dates);
      allY.push.apply(allY, chartData.close);
    }

    if (forecastPath && forecastPath.length > 0) {
      const fcDates = forecastPath.map(function (d) { return d.forecast_date; });
      const fcPrices = forecastPath.map(function (d) { return d.predicted_price || d.ensemble_prediction; });
      const fcLower = forecastPath.map(function (d) { return d.lower_bound; });
      const fcUpper = forecastPath.map(function (d) { return d.upper_bound; });
      allDates.push.apply(allDates, fcDates);
      allY.push.apply(allY, fcPrices);
      allY.push.apply(allY, fcLower);
      allY.push.apply(allY, fcUpper);

      traces.push({
        x: fcDates,
        y: fcUpper,
        type: 'scatter',
        mode: 'lines',
        name: 'Upper',
        line: { color: 'rgba(63, 185, 80, 0.5)', width: 1, dash: 'dot' },
      });
      traces.push({
        x: fcDates,
        y: fcLower,
        type: 'scatter',
        mode: 'lines',
        name: 'Lower',
        fill: 'tonexty',
        fillcolor: 'rgba(63, 185, 80, 0.15)',
        line: { color: 'rgba(63, 185, 80, 0.5)', width: 1, dash: 'dot' },
      });
      traces.push({
        x: fcDates,
        y: fcPrices,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Forecast',
        line: { color: '#3fb950', width: 2.5 },
        marker: { size: 5 },
      });
    }

    if (traces.length === 0) return;

    const tickFmt = chartTickFormat(refPrice);
    const yaxis = {
      gridcolor: '#21262d',
      showgrid: true,
      tickformat: tickFmt,
      autorange: true,
      fixedrange: false,
      zeroline: false,
    };
    if (isMicroPrice) {
      yaxis.exponentformat = 'e';
      yaxis.dtick = null;
    }

    Plotly.newPlot(container, traces, {
      margin: { t: 30, r: 20, b: 50, l: 60 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#8b949e', size: 11 },
      xaxis: {
        gridcolor: '#21262d',
        showgrid: true,
        tickformat: '%b %d',
        rangeslider: { visible: false },
      },
      yaxis: yaxis,
      showlegend: traces.length > 1,
      legend: { x: 0, y: 1.08, orientation: 'h' },
    });
  }

  function renderEvaluation(data) {
    const bestEl = document.getElementById('eval-best');
    const metricsEl = document.getElementById('eval-metrics');

    const evalData = data && data.evaluation ? data.evaluation : data;
    if (!evalData || !evalData.best_model) {
      bestEl.textContent = 'No evaluation data for this coin.';
      metricsEl.classList.add('hidden');
      return;
    }
    const m = evalData.metrics || (evalData.mae != null ? {
      mae: evalData.mae,
      rmse: evalData.rmse,
      mape: evalData.mape_pct != null ? evalData.mape_pct : evalData.mape,
      directional_accuracy: evalData.directional_accuracy_pct != null ? evalData.directional_accuracy_pct : evalData.directional_accuracy,
    } : null);
    if (!m) {
      metricsEl.classList.add('hidden');
      return;
    }
    bestEl.textContent = 'Best model: ' + evalData.best_model;
    metricsEl.innerHTML = '';
    const rows = [
      ['MAE', m.mae],
      ['RMSE', m.rmse],
      ['MAPE', formatPercent(m.mape || m.mape_pct)],
      ['Dir. acc.', formatPercent(m.directional_accuracy || m.directional_accuracy_pct)],
    ];
    rows.forEach(function (r) {
      const span = document.createElement('span');
      span.innerHTML = '<strong>' + formatNumber(r[1]) + '</strong>' + r[0];
      metricsEl.appendChild(span);
    });
    metricsEl.classList.remove('hidden');
  }

  function clearRefreshTimer() {
    if (refreshTimer) {
      clearInterval(refreshTimer);
      refreshTimer = null;
    }
  }

  function scheduleRefresh() {
    clearRefreshTimer();
    if (!autoRefreshToggle || !autoRefreshToggle.checked) return;
    const sec = parseInt(refreshIntervalInput ? refreshIntervalInput.value : DEFAULT_REFRESH_SEC, 10);
    const ms = Math.min(600000, Math.max(15000, (isNaN(sec) ? DEFAULT_REFRESH_SEC : sec) * 1000));
    refreshTimer = setInterval(function () {
      if (coinSelect && coinSelect.value) loadData();
    }, ms);
  }

  async function loadData() {
    const coin = (coinSelect && coinSelect.value) ? coinSelect.value.trim() : '';
    if (!coin) return;

    hideAll();
    loadingEl.textContent = 'Loading…';
    loadingEl.classList.remove('hidden');

    try {
      const [pathRes, chartRes, evalRes] = await Promise.all([
        fetch(API_BASE + '/api/forecast-path/' + encodeURIComponent(coin)),
        fetch(API_BASE + '/api/chart/' + encodeURIComponent(coin) + '?days=90'),
        fetch(API_BASE + '/api/evaluation/' + encodeURIComponent(coin)),
      ]);

      hideAll();

      let pred = null;
      let chart = null;
      let evalData = null;

      if (pathRes.ok) {
        pred = await pathRes.json();
      } else {
        const fallback = await fetch(API_BASE + '/predictions/' + encodeURIComponent(coin));
        if (fallback.ok) pred = await fallback.json();
      }

      if (!pred) {
        const body = pathRes.ok ? {} : await pathRes.json().catch(function () { return {}; });
        showError((body.detail || pathRes.statusText) || 'Failed to load forecast');
        return;
      }

      if (chartRes.ok) chart = await chartRes.json();
      if (evalRes.ok) evalData = await evalRes.json();
      if (pred.evaluation) evalData = Object.assign({}, evalData || {}, pred.evaluation);

      const refPrice = pred.current_price || pred.average_prediction;
      const latestMarket = pred.latest_market_timestamp || (chart && chart.latest_market_date);

      renderPrediction(pred);
      predictionEl.classList.remove('hidden');

      if (pred.forecast_path && pred.forecast_path.length > 0) {
        renderForecastTable(pred.forecast_path, refPrice);
        forecastTableSection.classList.remove('hidden');
      } else {
        forecastTableSection.classList.add('hidden');
      }

      if (chart || (pred.forecast_path && pred.forecast_path.length > 0)) {
        renderChart(chart, pred.forecast_path, refPrice, latestMarket);
        chartSection.classList.remove('hidden');
      }

      if (evalData || pred.evaluation) {
        renderEvaluation(pred.evaluation ? pred : evalData);
        evaluationSection.classList.remove('hidden');
      }

      scheduleRefresh();
    } catch (err) {
      showError('Error loading data: ' + (err.message || err));
    }
  }

  if (coinSelect) {
    coinSelect.addEventListener('change', function () {
      if (fetchBtn) fetchBtn.disabled = !coinSelect.value;
      clearRefreshTimer();
      if (coinSelect.value) scheduleRefresh();
    });
  }

  if (fetchBtn) fetchBtn.addEventListener('click', loadData);

  if (autoRefreshToggle) {
    autoRefreshToggle.addEventListener('change', function () {
      if (autoRefreshToggle.checked && coinSelect && coinSelect.value) scheduleRefresh();
      else clearRefreshTimer();
    });
  }

  if (refreshIntervalInput) {
    refreshIntervalInput.addEventListener('change', function () {
      if (autoRefreshToggle && autoRefreshToggle.checked) scheduleRefresh();
    });
  }

  loadCoins();
})();
