let STOCK_SET = null;

async function analyzeWSB() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const dashboard = document.getElementById('dashboard');

    // disable button
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Loading...';
    dashboard.style.display = 'none';

    try {
        // load in the stock lists to use for matching (s&p 500 stocks)
        if (STOCK_SET == null) {
            STOCK_SET = new Set(await fetch('stocks.json')
                .then(response => response.json()))
        }

        const posts = await getPosts();
        const analysis = await processPosts(posts);
        display(analysis);

        dashboard.style.display = 'grid';
    } catch (e) {
        console.error(e);
    } finally {
        // re-enable button
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Run Analysis';
    }
}

async function getPosts() {
    const postLimit = parseInt(document.getElementById('postLimit').value) || 5;
    const sortBy = document.getElementById('sortBy').value;
    const timeframe = document.getElementById('timeframe').value;

    // call the backend
    let url = `http://127.0.0.1:3500/api/reddit?limit=${postLimit}&sort=${sortBy}`;
    if (sortBy === 'top') url += `&t=${timeframe}`;

    const response = await fetch(url);
    const data = await response.json();

    // get only the relevant data
    return data.data.children.map(post => ({
        title: post.data.title,
        content: post.data.selftext || '',
        score: post.data.score,
        comments: post.data.num_comments
    }));
}

async function processPosts(posts) {
    const stockMentions = findTickers(posts);
    const sentiments = await analyzeSentiment(posts)

    const sentimentCounts = { bullish: 0, bearish: 0, neutral: 0 };
    let totalConfidence = 0, sentimentScoreSum = 0;

    sentiments.forEach(sentiment => {
        sentimentCounts[sentiment.label]++;
        totalConfidence += sentiment.confidence;
        sentimentScoreSum += sentiment.score;
    });

    const avgConfidence = (totalConfidence / sentiments.length * 100).toFixed(1);
    const avgSentiment = parseFloat((sentimentScoreSum / sentiments.length).toFixed(3));

    return { stockMentions, sentimentCounts, sentiments, avgConfidence: `${avgConfidence}%`, totalPosts: posts.length, uniqueStocks: Object.keys(stockMentions).length, avgSentiment, lastUpdated: new Date().toLocaleString() };
}

async function analyzeSentiment(posts) {
    try {
        // the model has a limit of 512 chars for input text
        const texts = posts.map(post => (post.title + ' ' + post.content).substring(0, 512));

        // call backend for sentiment analysis
        const response = await fetch(`http://127.0.0.1:3500/api/sentiment`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ inputs: texts })
        });

        const results = await response.json();
        return results.map((result, i) => ({
            label: result.label.toLowerCase(),
            confidence: result.score,
            score: result.label === 'BULLISH' ? result.score : result.label === 'BEARISH' ? -result.score : 0,
            post: posts[i]
        }));
    } catch {
        // fallback to neutral if something fails
        return posts.map(post => ({
            label: 'neutral',
            confidence: 0.5,
            score: 0,
            post
        }));
    }

}

function findTickers(posts) {
    const mentions = {};

    posts.forEach(post => {
        let text = (post.title + ' ' + post.content).toUpperCase();

        // match for $TICKER format
        const words = text.match(/[$]?[A-Z.]{1,5}\b/g);
        if (!words) return;

        words.forEach(rawWord => {
            const hasDollar = rawWord.startsWith('$');
            let ticker = hasDollar ? rawWord.slice(1) : rawWord;

            // was getting a lot of fale positives with common/everyday words
            if (ticker.length <= 2 && !hasDollar) {
                return
            }

            // only look at tickers in s&p
            if (STOCK_SET.has(ticker)) {
                mentions[ticker] = (mentions[ticker] || 0) + 1;
            }
        });
    });

    return Object.fromEntries(
        Object.entries(mentions).sort(([, a], [, b]) => b - a)
    );
}


function display(analysis) {
    let { stockMentions, sentimentCounts, sentiments, avgConfidence, totalPosts, uniqueStocks, avgSentiment, lastUpdated } = analysis

    const total = sentimentCounts.bullish + sentimentCounts.bearish + sentimentCounts.neutral;

    const ele = document.getElementById('overallSentiment');
    if (sentimentCounts.bullish > sentimentCounts.bearish) {
        ele.textContent = 'Bullish 📈';
        ele.className = 'metric-value sentiment-positive';
    } else if (sentimentCounts.bearish > sentimentCounts.bullish) {
        ele.textContent = 'Bearish 📉';
        ele.className = 'metric-value sentiment-negative';
    } else {
        ele.textContent = 'Neutral ➡️';
        ele.className = 'metric-value sentiment-neutral';
    }

    document.getElementById('bullishCount').textContent = `${sentimentCounts.bullish} (${((sentimentCounts.bullish / total) * 100).toFixed(1)}%)`;
    document.getElementById('bearishCount').textContent = `${sentimentCounts.bearish} (${((sentimentCounts.bearish / total) * 100).toFixed(1)}%)`;
    document.getElementById('neutralCount').textContent = `${sentimentCounts.neutral} (${((sentimentCounts.neutral / total) * 100).toFixed(1)}%)`;
    document.getElementById('aiConfidence').textContent = avgConfidence;

    // list of top stocks
    const stockList = document.getElementById('stockList');
    stockList.innerHTML = ''; // build the list
    for (const [stock, count] of Object.entries(stockMentions)) {
        const stockItem = document.createElement('div');
        stockItem.className = 'stock-item';
        stockItem.innerHTML = `<span class="stock-symbol">$${stock}</span><span class="mention-count">${count}</span>`;
        stockList.appendChild(stockItem);
    }

    // the analysis summary
    document.getElementById('postsAnalyzed').textContent = totalPosts;
    document.getElementById('uniqueStocks').textContent = uniqueStocks;
    document.getElementById('avgSentiment').textContent = avgSentiment;
    document.getElementById('lastUpdated').textContent = lastUpdated;

    // recent posts 
    const recentPostsContainer = document.getElementById('recentPosts');
    recentPostsContainer.innerHTML = '';

    for (const sentiment of sentiments.slice(0, 10)) {
        const postItem = document.createElement('div');
        postItem.className = `post-item ${sentiment.label}`;
        postItem.innerHTML = `
        <div class="post-title">${sentiment.post.title}</div>
        <div class="post-meta">
            <span>👍 ${sentiment.post.score}</span>
            <span>💬 ${sentiment.post.comments}</span>
            <span>📊 ${sentiment.label} (${(sentiment.confidence * 100).toFixed(1)}%)</span>
        </div>`;

        recentPostsContainer.appendChild(postItem);
    }
}
