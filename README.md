# Guia Completo: Como Acompanhar Pesquisa em IA

> Última atualização: Fevereiro 2026

---

## 1. Fontes Primárias de Papers e Pesquisa

### Repositórios de Preprints (Acesso Direto à Pesquisa)

| Plataforma | URL | O que encontrar |
|---|---|---|
| **arXiv** | [arxiv.org/list/cs.AI/recent](https://arxiv.org/list/cs.AI/recent) | A fonte #1 mundial de preprints em IA. Categorias-chave: `cs.AI`, `cs.LG`, `cs.CL`, `cs.CV` |
| **Papers With Code** | [paperswithcode.com](https://paperswithcode.com) | Papers + código-fonte + benchmarks. Mostra trending papers e SOTA por tarefa |
| **Hugging Face Papers** | [huggingface.co/papers](https://huggingface.co/papers) | Curadoria diária de papers relevantes com discussões da comunidade |
| **alphaXiv** | [alphaxiv.org](https://www.alphaxiv.org) | Papers do arXiv com resumos simplificados e discussões por blog |
| **Arxiv Sanity** | [arxiv-sanity-lite.com](http://arxiv-sanity-lite.com) | Criado por Andrej Karpathy. Recomendações personalizadas baseadas nos seus interesses |
| **Cool Papers** | [papers.cool/arxiv/cs.AI](https://papers.cool/arxiv/cs.AI) | Interface visual imersiva para descoberta de papers por área |
| **DAIR.AI ML Papers of the Week** | [github.com/dair-ai/ML-Papers-of-the-Week](https://github.com/dair-ai/ML-Papers-of-the-Week) | Curadoria semanal dos papers mais impactantes. Excelente filtro de ruído |
| **Awesome AI Papers** | [github.com/aimerou/awesome-ai-papers](https://github.com/aimerou/awesome-ai-papers) | Lista curada classificada por impacto: Históricos (10k+ citações), Importantes (50+), e Tendências |

### Buscadores Acadêmicos

| Ferramenta | URL | Diferencial |
|---|---|---|
| **Semantic Scholar** | [semanticscholar.org](https://www.semanticscholar.org) | IA que entende semântica dos papers. 200M+ registros. API aberta |
| **Google Scholar** | [scholar.google.com](https://scholar.google.com) | O mais abrangente. Use operadores Boolean para refinar |
| **Consensus** | [consensus.app](https://consensus.app) | Busca em 200M+ papers e retorna respostas baseadas em evidências com citações |
| **Elicit** | [elicit.com](https://elicit.com) | Assistente de pesquisa com IA. Extrai dados, compara achados |
| **Connected Papers** | [connectedpapers.com](https://www.connectedpapers.com) | Grafo visual mostrando conexões entre papers. Ótimo para mapear um campo |
| **Research Rabbit** | [researchrabbit.ai](https://researchrabbit.ai) | Mapeamento de literatura visual. 100% gratuito. Descobre papers relacionados automaticamente |
| **Scite.ai** | [scite.ai](https://scite.ai) | Verifica se citações suportam, contradizem ou apenas mencionam um claim |
| **Ai2 Paper Finder** | [paper-finder.ai2.org](https://allenai.org/blog/paper-finder) | Busca inteligente do Allen AI. Aceita queries complexas em linguagem natural |

### Conferências Top (Onde os Breakthroughs São Publicados)

| Conferência | Foco | Quando |
|---|---|---|
| **NeurIPS** | ML/IA geral | Dezembro |
| **ICML** | Machine Learning | Julho |
| **ICLR** | Representações aprendidas | Maio |
| **CVPR** | Visão computacional | Junho |
| **ACL / EMNLP** | Processamento de linguagem natural | Julho / Novembro |
| **AAAI** | IA geral | Fevereiro |
| **ICAI (IEEE)** | IA aplicada (saúde, engenharia, negócios) | Variável |

### Journals de Alto Impacto

- **Nature Machine Intelligence** — Pesquisa interdisciplinar de ponta
- **Foundations and Trends in Machine Learning** — Maior fator de impacto em ML
- **Artificial Intelligence (Elsevier)** — Um dos mais antigos e respeitados
- **JMLR (Journal of Machine Learning Research)** — Open access, alta qualidade
- **Neural Networks** — Em publicação desde 1988, cobre redes neurais e sistemas de aprendizado
- **IEEE Transactions on PAMI** — Padrão-ouro em pattern recognition e ML
- **International Journal of Computer Vision** — Maior impacto em visão computacional

---

## 2. Chaves de Pesquisa para Usar em Buscadores Acadêmicos

### Termos Fundamentais (Use em Qualquer Buscador)

**LLMs e Modelos de Linguagem:**
- `"large language models"` / `"LLM"`
- `"transformer architecture"` / `"attention mechanism"`
- `"chain-of-thought reasoning"` / `"reasoning models"`
- `"reinforcement learning from human feedback"` / `"RLHF"`
- `"instruction tuning"` / `"fine-tuning"`
- `"mixture of experts"` / `"MoE"`
- `"sparse attention"` / `"efficient transformers"`

**IA Generativa:**
- `"diffusion models"` / `"text-to-image"`
- `"generative adversarial networks"` / `"GANs"`
- `"multimodal models"` / `"vision-language models"`
- `"text-to-video generation"`

**Agentes e Automação:**
- `"AI agents"` / `"agentic AI"`
- `"tool use"` / `"function calling"`
- `"autonomous agents"` / `"multi-agent systems"`
- `"code generation"` / `"AI coding assistants"`

**Segurança e Alinhamento:**
- `"AI alignment"` / `"AI safety"`
- `"red teaming"` / `"adversarial attacks"`
- `"hallucination detection"` / `"factual grounding"`
- `"constitutional AI"` / `"RLHF alternatives"`

**Infraestrutura e Escala:**
- `"scaling laws"` / `"compute efficiency"`
- `"model compression"` / `"quantization"`
- `"knowledge distillation"` / `"pruning"`
- `"distributed training"` / `"inference optimization"`

**Tópicos Emergentes (2025-2026):**
- `"world models"` / `"JEPA"`
- `"test-time compute"` / `"inference-time scaling"`
- `"synthetic data"` / `"data-centric AI"`
- `"small language models"` / `"on-device AI"`
- `"retrieval augmented generation"` / `"RAG"`
- `"model context protocol"` / `"MCP"`

### Operadores Boolean para Google Scholar

```
# Busca exata
"large language models" AND "reinforcement learning"

# Excluir ruído
"transformer" AND "attention" -survey -review

# Por período
"AI agents" after:2024

# Por autor
author:"Yann LeCun" "world models"

# Por venue
source:"NeurIPS" "reasoning"
```

### Categorias arXiv Mais Relevantes

| Código | Área |
|---|---|
| `cs.AI` | Artificial Intelligence |
| `cs.LG` | Machine Learning |
| `cs.CL` | Computation and Language (NLP) |
| `cs.CV` | Computer Vision |
| `cs.RO` | Robotics |
| `cs.SE` | Software Engineering |
| `stat.ML` | Machine Learning (Statistics) |

---

## 3. Pesquisadores e Líderes em IA para Acompanhar

### Os "Godfathers" e Pioneiros

| Nome | Afiliação | Área | X (Twitter) | Outras Redes |
|---|---|---|---|---|
| **Geoffrey Hinton** | Independente (ex-Google) | Deep learning, AI Safety | [@geoffreyhinton](https://x.com/geoffreyhinton) | Nobel de Física 2024, Turing Award 2018 |
| **Yann LeCun** | Meta AI / NYU | CNNs, world models, self-supervised learning | [@ylecun](https://x.com/ylecun) | Turing Award 2018. Muito ativo no X com debates técnicos |
| **Yoshua Bengio** | Mila / U. Montreal | Deep learning, AI safety, generative flow networks | [@yoshaboris](https://x.com/yoshaboris) | Turing Award 2018 |
| **Andrew Ng** | Stanford / DeepLearning.AI / Landing AI | ML educação, data-centric AI | [@AndrewYNg](https://x.com/AndrewYNg) | [LinkedIn](https://linkedin.com/in/andrewyng), Newsletter "The Batch" |
| **Fei-Fei Li** | Stanford HAI | Visão computacional, AI for social good | [@drfeifei](https://x.com/drfeifei) | Uma das pesquisadoras mais citadas do mundo |

### Builders e Pesquisadores Ativos

| Nome | Afiliação | Área | X (Twitter) | Outras Redes |
|---|---|---|---|---|
| **Andrej Karpathy** | Eureka Labs (fundador) | Deep learning, educação em IA, vibe coding | [@karpathy](https://x.com/karpathy) | [YouTube](https://youtube.com/@andrejkarpathy), [karpathy.ai](https://karpathy.ai). Melhor educador técnico em IA |
| **Ilya Sutskever** | Safe Superintelligence Inc. (SSI) | AGI, AI safety | [@ilyasut](https://x.com/ilyasut) | Co-fundador OpenAI. Posts raros mas impactantes |
| **Demis Hassabis** | Google DeepMind | AGI, AlphaFold, raciocínio | [@demaboris](https://x.com/demishassabis) | Nobel de Química 2024 |
| **Dario Amodei** | Anthropic (CEO) | AI safety, scaling | [@DarioAmodei](https://x.com/DarioAmodei) | Co-fundador da Anthropic |
| **Sam Altman** | OpenAI (CEO) | LLMs, AGI, produto | [@sama](https://x.com/sama) | Figura polarizadora, mas central no ecossistema |
| **Jensen Huang** | NVIDIA (CEO) | Hardware para IA, GPUs, infraestrutura | Sem conta X ativa | [LinkedIn](https://linkedin.com/in/jenhsunhuang). Keynotes da NVIDIA são essenciais |
| **Jeremy Howard** | fast.ai / answer.ai | ML prático, democratização de IA | [@jeaboris](https://x.com/jeremyphoward) | [fast.ai](https://fast.ai). Cursos gratuitos de altíssima qualidade |
| **Sebastian Raschka** | Lightning AI | LLMs, ML prático, educação | [@rasaboris](https://x.com/rasaboris) | [Newsletter no Substack](https://magazine.sebastianraschka.com). Lista curada de papers de LLMs |
| **Chris Olah** | Anthropic | Interpretabilidade de redes neurais | [@ch402](https://x.com/ch402) | [colah.github.io](https://colah.github.io). Blog icônico de visualização de conceitos de ML |
| **Lex Fridman** | MIT | Entrevistas com líderes de IA | [@lexfridman](https://x.com/lexfridman) | [YouTube](https://youtube.com/@lexfridman). Podcast essencial |
| **Aravind Srinivas** | Perplexity (CEO) | AI search, information retrieval | [@AravSrinivas](https://x.com/AravSrinivas) | Construindo o futuro da busca com IA |
| **Thomas Wolf** | Hugging Face (co-fundador) | Open source AI, NLP | [@Thom_Wolf](https://x.com/Thom_Wolf) | Central no movimento de IA open source |

### Labs de Pesquisa Corporativa — Blogs Oficiais

Estes blogs publicam pesquisa de ponta, muitas vezes antes dos papers formais:

| Lab | Blog/URL |
|---|---|
| **OpenAI** | [openai.com/research](https://openai.com/research) |
| **Google DeepMind** | [deepmind.google/research](https://deepmind.google/research) |
| **Anthropic** | [anthropic.com/research](https://www.anthropic.com/research) |
| **Meta AI (FAIR)** | [ai.meta.com/research](https://ai.meta.com/research) |
| **Google AI** | [ai.google/research](https://ai.google/research) |
| **Microsoft Research** | [microsoft.com/en-us/research/lab/ai](https://www.microsoft.com/en-us/research/) |
| **NVIDIA Research** | [research.nvidia.com](https://research.nvidia.com) |
| **DeepSeek** | [github.com/deepseek-ai](https://github.com/deepseek-ai) |
| **Allen AI (Ai2)** | [allenai.org/blog](https://allenai.org/blog) |

---

## 4. Sistema Completo para Ficar Atualizado

### Newsletters (Diárias/Semanais)

| Newsletter | Frequência | Foco | Link |
|---|---|---|---|
| **The Batch** (Andrew Ng) | Semanal | Pesquisa + indústria, acessível | [deeplearning.ai/the-batch](https://www.deeplearning.ai/the-batch/) |
| **The Rundown AI** | Diária | Notícias práticas, 1M+ assinantes | [therundown.ai](https://www.therundown.ai/) |
| **The Neuron** | Diária | Trends + ferramentas, 3 min de leitura | [theneurondaily.com](https://www.theneurondaily.com/) |
| **Import AI** (Jack Clark) | Semanal | Papers + política de IA, profundo | [importai.net](https://importai.net/) |
| **Sebastian Raschka's Magazine** | Semanal | Papers de LLM organizados por tópico | [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/) |
| **Latent Space** | Semanal | Para AI engineers, muito técnico | [latent.space](https://www.latent.space/) |
| **Davis Summarizes Papers** | Semanal | Resumos de papers | Substack |
| **AI Weekly Digest** (Last Week in AI) | Semanal | Resumo abrangente da semana | [lastweekin.ai](https://lastweekin.ai/) |

### Podcasts Essenciais

| Podcast | Hosts | Foco | Onde ouvir |
|---|---|---|---|
| **Lex Fridman Podcast** | Lex Fridman | Entrevistas longas com líderes de IA | YouTube, Spotify, Apple |
| **No Priors** | Sarah Guo & Elad Gil | IA + negócios + investimento | Spotify, Apple |
| **Latent Space** | Alessio Fanelli & Swyx | Técnico, para AI engineers | Spotify, Apple |
| **TWIML AI** | Sam Charrington | ML/AI pesquisa + aplicações | Spotify, Apple |
| **Practical AI** | Chris Benson & Daniel Whitenack | ML acessível e prático | Spotify, Apple |
| **Eye on AI** | Craig S. Smith (ex-NYT) | Visão ampla + contexto global | Spotify, Apple |
| **The Cognitive Revolution** | Nathan Labenz & Erik Torenberg | AGI, policy, ética | Spotify, Apple |
| **High Agency** | Raza Habib (Humanloop) | Construção de produtos com LLMs | Spotify, Apple |
| **Last Week in AI** | Andrey Kurenkov & Harris | Resumo semanal de notícias de IA | Spotify, Apple |
| **AI Daily Brief** | NLW | Análise diária do cenário de IA | Apple Podcasts |

### Canais de YouTube Técnicos

| Canal | Conteúdo |
|---|---|
| **Andrej Karpathy** | Tutoriais profundos sobre neural networks, os melhores da internet |
| **3Blue1Brown** | Visualizações matemáticas de ML/deep learning |
| **Yannic Kilcher** | Explicações detalhadas de papers recentes |
| **Two Minute Papers** | Resumos rápidos de pesquisa nova |
| **AI Explained** | Análise equilibrada de desenvolvimentos em IA |

### Comunidades e Fóruns

| Plataforma | Link | Valor |
|---|---|---|
| **Hugging Face Discord** | Via huggingface.co | Comunidade open source de ML |
| **r/MachineLearning** | reddit.com/r/MachineLearning | Discussões técnicas de papers |
| **r/LocalLLaMA** | reddit.com/r/LocalLLaMA | Deployment local de LLMs (alinhado com seus projetos) |
| **Latent Space Discord** | Via latent.space | Comunidade de AI engineers |
| **ResearchGate** | researchgate.net | Rede social para pesquisadores |

---

## 5. Rotina Sugerida para se Manter Atualizado

### Diário (~15 min)
1. Ler **The Neuron** ou **The Rundown AI** (newsletter rápida)
2. Checar **Papers With Code** trending

### Semanal (~1-2h)
1. Ler **DAIR.AI ML Papers of the Week** no GitHub
2. Ouvir 1 episódio de podcast (Lex Fridman ou Latent Space)
3. Ler **The Batch** de Andrew Ng
4. Checar blogs de 2-3 labs corporativos (OpenAI, DeepMind, Anthropic)

### Mensal (~3-4h)
1. Deep-dive em 2-3 papers usando **Semantic Scholar** ou **Connected Papers**
2. Revisar proceedings da conferência principal do mês
3. Atualizar lista de pesquisadores que acompanha no X

### Setup Recomendado
- **X (Twitter)**: Crie uma lista dedicada com os pesquisadores da tabela acima. Use a lista no lugar do feed algorítmico
- **RSS**: Use Feedly ou similar para acompanhar blogs dos labs
- **Alertas**: Configure Google Scholar Alerts para seus termos de interesse
- **Arxiv Sanity**: Crie perfil e marque papers relevantes para receber recomendações personalizadas

---

## 6. Dica Extra: Ferramentas de IA para Pesquisa

Para acelerar a leitura de papers, considere:

- **Consensus** (consensus.app) — Melhor para respostas rápidas baseadas em evidência. Free: 10 buscas/mês
- **Scite.ai** — Verifica credibilidade de citações. Pro: $10/mês
- **Research Rabbit** — Mapeamento visual de literatura. 100% gratuito
- **Semantic Scholar** — API aberta, ótima para integrar com seus projetos Python
- **Perplexity** (perplexity.ai) — Busca conversacional com citações acadêmicas

---

*Este guia foi compilado a partir de múltiplas fontes verificadas. Para a versão mais atualizada das listas de pesquisadores e ferramentas, consulte a lista TIME100 AI (time.com) e o relatório Favikon AI200.*
