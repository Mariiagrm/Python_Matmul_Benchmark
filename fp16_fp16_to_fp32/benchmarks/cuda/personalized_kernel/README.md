# SGEMM optimizado para Ada Lovelace (RTX 4090)

Kernel CUDA de producto matriz-matriz en precisión simple que implementa todas
las optimizaciones discutidas en el capítulo correspondiente del TFG sobre la
arquitectura **Ada Lovelace** (compute capability **8.9**). Valida y compara
sus resultados contra **cuBLAS** (`cublasSgemm`).

## Compilación y ejecución

```bash
make            # compila con nvcc -O3 -arch=sm_89 -use_fast_math
./sgemm_ada     # ejecuta con M=N=K=4096 por defecto
./sgemm_ada 8192 8192 8192   # tamaños personalizados (múltiplos de 128)
```

La flag `--ptxas-options=-v` que incluye el Makefile imprime durante la
compilación el número exacto de **registros por hilo** y de **bytes de
memoria compartida por bloque** que utiliza `ptxas`. Esos números permiten
calcular la ocupación real (sección de registros del TFG) sin depender de
estimaciones.

## Correspondencia entre el código y el TFG

| Optimización del TFG | Implementación en `sgemm_ada.cu` |
|---|---|
| Tile de salida de **128×128** | `#define BM 128`, `#define BN 128` |
| **256 hilos/bloque** en rejilla **16×16** | `BLOCK_DIM_X = BLOCK_DIM_Y = 16` |
| Sub-tile **8×8** por hilo en registros | `TM = TN = 8`, array `c_reg[TM][TN]` |
| **Doble buffer** en memoria compartida | `__shared__ float As[2][...]`, `Bs[2][...]` |
| **Pipeline asíncrono (`cp.async`)** | `__pipeline_memcpy_async`, `__pipeline_commit`, `__pipeline_wait_prior` |
| Cargas **`float4`** desde global | Conversión vía `reinterpret_cast<float4*>` y `cp.async` de 16 B |
| **Padding** anti bank-conflict | `BN_PAD = 4`, fila de `Bs` con `BN_STRIDE = 132` |
| Compilación con `-arch=sm_89 -use_fast_math` | En el `Makefile` |
| Limitar registros para mantener 2–3 bloques/SM | `__launch_bounds__(256, 2)` |
| Escritura vectorizada de C | `float4` vía `reinterpret_cast` |
| Validación frente a cuBLAS | `cublas_sgemm_rm()` + `max_abs_diff()` |
| Benchmark relativo | Sección `// ---- Benchmark ----` |

## Esquema del pipeline

```
   Prólogo:  cp.async  ->  buf[0]            commit
   Iter t :  cp.async  ->  buf[(t+1)%2]      commit
             wait_prior(1)   // espera buf[t%2]
             __syncthreads()
             compute(buf[t%2])
   Epílogo: wait_prior(0); compute(buf[última])
```

Se mantienen dos `commit` simultáneos en vuelo. `wait_prior(1)` significa
*“espera hasta que quede como mucho un commit pendiente”*, es decir, el más
reciente sigue copiando mientras se computa sobre el anterior.

## Notas sobre el diseño

**Layout de `As` (transpuesto).** `A` es M×K en *row-major*; al cargarlo a
memoria compartida se transpone para que el bucle interno `a_reg[i] = As[k][ty*TM + i]`
realice accesos secuenciales en la dimensión M, lo que minimiza conflictos de
banco y favorece la coalescencia interna. Por construcción, los cuatro destinos
de un `float4` quedan en cuatro filas distintas de `As`, por lo que la
transposición se implementa con cuatro `cp.async` de 4 bytes en lugar de uno
de 16. Para `Bs` no hay transposición y se emite una única `cp.async` de 16 B.

**Padding en `Bs`.** El acceso `Bs[k][tx*TN + j]` con `TN = 8` provoca un
patrón de bancos con período 4 (4-way bank conflict potencial). Añadir cuatro
elementos de padding por fila desplaza la cabecera de cada fila siguiente y
mitiga el conflicto en la mayoría de patrones; una eliminación completa
requeriría *swizzling* XOR, que se omite por claridad pedagógica.

**Ocupación real.** Con `__launch_bounds__(256, 2)` el compilador limita los
registros por hilo hasta permitir al menos dos bloques residentes por SM, lo
que equivale a 16 warps residentes y ~33 % de ocupación. Si `ptxas` reporta
~80 reg/hilo (caso típico), se obtienen tres bloques residentes ⇒ 24 warps ⇒
**~50 % de ocupación**, el valor citado en el TFG.

## Limitaciones conscientes

- No se procesan bordes: `M`, `N` deben ser múltiplos de 128 y `K` de 8.
  Añadir manejo de bordes complica el kernel sin aportar valor pedagógico
  para el TFG; en un kernel de producción se delegan los bordes a un
  *cleanup kernel* o se usa `cp.async` predicado.
- FP32 con CUDA cores; **no** se emplean Tensor Cores. La descripción del
  TFG con sub-tile `8×8` por hilo encaja con el modelo de CUDA cores; la
  variante con Tensor Cores requiere otro esquema basado en `wmma::` o
  PTX `mma.sync`, con fragmentos fijos de 16×16×16 distribuidos entre los
  hilos del warp.
- *Bank conflicts* mitigados, no eliminados. Una versión libre de
  conflictos requiere swizzling XOR.

## Resultado esperado en una RTX 4090

Salida típica para 4096×4096×4096:

```
SGEMM  M=4096  N=4096  K=4096  (tile 128x128, BK=8, thread 8x8)
Max |diff| frente a cuBLAS: 2.xxx e-04 [OK]
Kernel propio :  ~9.5 ms     ~14500 GFLOP/s
cuBLAS        :  ~6.0 ms     ~22900 GFLOP/s
Ratio propio / cuBLAS: ~63 %
```

Conforme al criterio definido en el TFG, un *ratio* del orden del **60–70 %**
de cuBLAS se considera un resultado excepcional para un kernel manuscrito en
FP32 sin Tensor Cores.
