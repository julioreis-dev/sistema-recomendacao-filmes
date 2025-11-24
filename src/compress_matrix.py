# compress_matrix.py
import numpy as np
import os

print("📦 Compactando matriz de similaridade...")

# Carregar matriz original
cosine_sim = np.load('model_artifacts/cosine_sim_matrix.npy')
print(f"🔢 Matriz original: {cosine_sim.shape}")
print(f"💾 Tamanho original: {cosine_sim.nbytes / (1024*1024):.2f} MB")

# Salvar comprimido
np.savez_compressed('model_artifacts/cosine_sim_matrix_compressed.npz', matrix=cosine_sim)

# Verificar tamanho comprimido
compressed_size = os.path.getsize('model_artifacts/cosine_sim_matrix_compressed.npz') / (1024*1024)
print(f"📦 Tamanho comprimido: {compressed_size:.2f} MB")
print(f"📊 Redução: {(1 - compressed_size/(cosine_sim.nbytes/(1024*1024)))*100:.1f}%")

print("✅ Compactação concluída!")