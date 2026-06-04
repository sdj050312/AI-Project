window.onload = function () {

    const predList = window.PRED_LIST;
    const livePrice = window.LIVE_PRICE;

    if (!predList || predList.length === 0) {
        console.log("No data");
        return;
    }

    const lastPred = predList[predList.length - 1];

    let signal = "";
    let sellPrice = 0;

    if (lastPred < -0.005) {
        signal = "🔴 SELL";
        sellPrice = livePrice * 0.98;
    }
    else if (lastPred > 0.005) {
        signal = "🟢 BUY";
        sellPrice = livePrice * 1.02;
    }
    else {
        signal = "🟡 HOLD";
        sellPrice = livePrice;
    }

    const profitRate = ((sellPrice - livePrice) / livePrice) * 100;

    document.getElementById("signal").innerText = signal;
    document.getElementById("sellPrice").innerText = sellPrice.toFixed(2);
    document.getElementById("profitRate").innerText = profitRate.toFixed(2) + "%";
}