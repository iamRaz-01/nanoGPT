# nanoGPT University Curriculum

Welcome to the **nanoGPT University Curriculum**! This course is designed to take you from a basic understanding of machine learning to deep, first-principles mastery of modern generative language models, PyTorch optimizations, and distributed GPU training.

Each lesson is written as a detailed, local markdown file (`README.md`) containing theory, visual diagrams, mathematical derivations, code walkthroughs, and practical exercises.

---

## Course Syllabus

### Phase 1: Foundations & Architecture
1. **[Lesson 1: Introduction to GPT, Transformers, and nanoGPT](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_1_intro.md)**
   - Sequential processing bottlenecks of RNNs/LSTMs, the self-attention revolution, and decoder-only architecture.
2. **[Lesson 2: Repository Architecture and Execution Flow](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_2_architecture.md)**
   - End-to-end tracing of `python train.py` from initialization to the first optimization step.
3. **[Lesson 3: The Data Pipeline & Memory Mapping](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_3_data_pipeline.md)**
   - High-throughput dataset prep, `np.memmap` buffer management, pinned memory, and asynchronous transfer.
4. **[Lesson 4: Tokenization & Byte-Pair Encoding (BPE)](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_4_tokenization.md)**
   - BPE algorithms, GPT-2 vocabs, vocabulary padding for efficiency, and byte-level fallbacks.
5. **[Lesson 5: Embedding Layers (Token & Positional Embeddings)](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_5_embeddings.md)**
   - Token lookup, absolute vs relative positional embeddings, scaling laws, and modern alternatives (RoPE).
6. **[Lesson 6: Layer Normalization (LayerNorm vs RMSNorm)](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_6_layernorm.md)**
   - Math of LayerNorm, batch norm vs layer norm, bias/weight settings, and RMSNorm performance advantages.

### Phase 2: Core Attention & Blocks
7. **[Lesson 7: Attention Mechanism (Theory & Dot-Product)](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_7_attention_theory.md)**
   - Queries, Keys, Values, dot-product scaling, vanishing gradients in softmax, and causal masking.
8. **[Lesson 8: Multi-Head Causal Self-Attention](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_8_multi_head_attention.md)**
   - Tensor splitting, dimension transpositions, projections, MHA vs MQA vs GQA.
9. **[Lesson 9: Feed-Forward Networks & GELU](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_9_mlp_gelu.md)**
   - Channel-mixing MLP, GELU activation derivation, and SwiGLU modern architectures.
10. **[Lesson 10: The Transformer Block & Residual Connections](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_10_transformer_block.md)**
    - Pre-LN vs Post-LN gradient flow, residual scaled initialization, and block forward pass.
11. **[Lesson 11: The GPT Model class & Weight Tying](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_11_gpt_class.md)**
    - Wrapping it all together, cross-entropy formulation, and weight sharing math.

### Phase 3: Training & Optimization
12. **[Lesson 12: Weight Initialization & Model Surgery](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_12_weight_init.md)**
    - Initialization standard deviations, model truncation, and context window pruning.
13. **[Lesson 13: Learning Rate Scheduler & Warmup](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_13_lr_scheduler.md)**
    - Cosine decay schedules, learning rate warmup, and scaling laws (Chinchilla).
14. **[Lesson 14: Optimization Mechanics (AdamW & Fused Optimizers)](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_14_optim_adamw.md)**
    - Weight decay partitioning (2D vs 1D parameters), AdamW update steps, fused kernels, and grad clipping.
15. **[Lesson 15: Mixed Precision (Autocast, GradScaler)](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_15_mixed_precision.md)**
    - FP16, BF16, FP32 representation, autocast namespaces, and numeric underflow/scaling.

### Phase 4: Scaling & Performance
16. **[Lesson 16: Distributed Training (DDP & torchrun)](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_16_distributed_ddp.md)**
    - Rank, local rank, world size, gradient communication overhead, and micro-batching.
17. **[Lesson 17: Performance Optimizations (FlashAttention & Compile)](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_17_performance_opt.md)**
    - Memory-bound vs compute-bound, kernel fusion, compilation graphs, and Model FLOPs Utilization (MFU).
18. **[Lesson 18: Inference & Sampling Tactics](file:///c:/Users/admin/Documents/Projects/nanoGPT/curriculum/lessons/lesson_18_inference_sampling.md)**
    - Autoregressive generation loops, temperature scaling, Top-K/Top-P sampling, and KV caching.

---

## Interactive Learning Flow

1. Read the current lesson README file.
2. Complete the coding challenges or read assignments.
3. Reply to the main assistant thread with your answers to the **Lesson Interview Questions**.
4. Once evaluated, the assistant will update the progress and write the next lesson file.
