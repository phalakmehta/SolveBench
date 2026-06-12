import { useState, useEffect } from 'react';
import './index.css';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch data from public folder
    fetch('./data.json')
      .then(res => {
        if (!res.ok) throw new Error('Data not ready yet');
        return res.json();
      })
      .then(json => {
        setData(json);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setError('Experiment is still running. Waiting for results...');
        // Refresh every 10 seconds if not ready
        const timer = setTimeout(() => window.location.reload(), 10000);
        return () => clearTimeout(timer);
      });
  }, []);

  if (error && !data) {
    return (
      <div className="loader">
        <div>{error}</div>
      </div>
    );
  }

  if (loading || !data) {
    return <div className="loader">Loading Results...</div>;
  }

  // Calculate metrics
  const results = Array.isArray(data) ? data : (data.results || []);
  
  // Aggregate by Architecture
  const archStats = {};
  let totalCost = 0;
  
  results.forEach(run => {
    const arch = run.architecture;
    if (!archStats[arch]) {
      archStats[arch] = { 
        correct: 0, 
        total: 0, 
        cost: 0, 
        time: 0, 
        tokens: 0,
        reasoningDepthSum: 0,
        stepCompletenessSum: 0,
        contradictionSum: 0,
        problems: [] 
      };
    }
    const stat = archStats[arch];
    stat.total += 1;
    if (run.is_correct) stat.correct += 1;
    
    stat.cost += run.metrics?.estimated_usd || 0;
    stat.time += run.metrics?.latency_seconds || 0;
    stat.tokens += run.metrics?.total_tokens || 0;
    stat.contradictionSum += run.metrics?.self_contradiction?.score || 0;
    
    // Only accumulate reasoning scores if correct (as incorrect parses often have 0s or skipped logic)
    if (run.is_correct && run.reasoning_scores) {
      stat.reasoningDepthSum += run.reasoning_scores.reasoning_depth || 0;
      stat.stepCompletenessSum += run.reasoning_scores.step_completeness || 0;
    }
    
    stat.problems.push(run);
    
    totalCost += run.metrics?.estimated_usd || 0;
  });

  const archs = Object.keys(archStats).sort();

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1 className="title">SolveBench Execution Dashboard</h1>
        <p className="subtitle">Real-time analysis of LLM Agent Architectures</p>
      </header>

      <div className="grid">
        <div className="card">
          <div className="card-title">Total Runs</div>
          <div className="stat-value">{results.length}</div>
        </div>
        <div className="card">
          <div className="card-title">Total Cost (USD)</div>
          <div className="stat-value">${totalCost.toFixed(4)}</div>
        </div>
        <div className="card">
          <div className="card-title">Best Architecture</div>
          <div className="stat-value">
            {archs.length > 0 ? archs.reduce((a, b) => 
              (archStats[a].correct / archStats[a].total) > (archStats[b].correct / archStats[b].total) ? a : b
            ) : 'N/A'}
          </div>
        </div>
      </div>

      <div className="chart-container">
        <h2 className="card-title" style={{ marginBottom: '2rem' }}>Accuracy by Architecture</h2>
        <div className="bar-chart">
          {archs.map(arch => {
            const stat = archStats[arch];
            const accuracy = ((stat.correct / stat.total) * 100).toFixed(1);
            return (
              <div className="bar-row" key={arch}>
                <div className="bar-label">{arch.replace('A', 'Arch ')}</div>
                <div className="bar-track">
                  <div 
                    className="bar-fill" 
                    style={{ width: `${accuracy}%` }}
                  >
                    {accuracy}%
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="chart-container">
        <h2 className="card-title" style={{ marginBottom: '2rem' }}>Architecture Performance & Quality</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', whiteSpace: 'nowrap' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '1rem', borderBottom: '1px solid var(--border)' }}>Architecture</th>
                <th style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>Accuracy</th>
                <th style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>Avg Cost</th>
                <th style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>Avg Latency</th>
                <th style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>Avg Tokens</th>
                <th style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>Quality (Depth/5)</th>
                <th style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>Contradiction (0-1)</th>
              </tr>
            </thead>
            <tbody>
              {archs.map(arch => {
                const stat = archStats[arch];
                const accuracy = ((stat.correct / stat.total) * 100).toFixed(1);
                // Compute averages for quality metrics based on correct runs only
                const avgDepth = stat.correct > 0 ? (stat.reasoningDepthSum / stat.correct).toFixed(2) : 'N/A';
                return (
                  <tr key={arch}>
                    <td style={{ padding: '1rem', borderBottom: '1px solid var(--border)' }}>{arch.replace('_', ' ')}</td>
                    <td style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)', fontWeight: 'bold' }}>
                      {accuracy}%
                    </td>
                    <td style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>
                      ${(stat.cost / stat.total).toFixed(4)}
                    </td>
                    <td style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>
                      {(stat.time / stat.total).toFixed(1)}s
                    </td>
                    <td style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>
                      {Math.round(stat.tokens / stat.total)}
                    </td>
                    <td style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>
                      {avgDepth}
                    </td>
                    <td style={{ textAlign: 'right', padding: '1rem', borderBottom: '1px solid var(--border)' }}>
                      {(stat.contradictionSum / stat.total).toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}

export default App;
