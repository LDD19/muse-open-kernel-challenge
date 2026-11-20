# Notes

- Prior relevant background: about 10 vLLM config files 
- Started the contest late, agent hit wall and had decided to do a repo sweep and work on top of them instead. 
- A sweep has also been ran on autoresearch harnesses and related papers. 
- Some personal thoughts: 
  - While the harness worked well enough overnight, the climb rate was far greater with a human-in-the-loop. Many loops would spend too much time doing A/B tests without meaninful gains. This is most likely an alignment issue. 
  - Model selection matters. The preferred combo was Fable 5.2 with Opus 5.5 subagents. The largest session spawned over 92 subagents with no visible incoherence.
  - Gemini was the worst and smartest model. Every new idea came with 9 more ideas that were either hallucinating or breaking rules. Not suitable for long horizon tasks. 
  - The main bottlenecks of this contest was the lack of time and compute. At peak hours, no 8xH200 node was readily available. Some efficiency improvements could've been done on the harness side too. Knowledge wasn't an issue as frontier models are already better than most experts. However, they lack the intuition.