# COMO APLICAR O FIX NO GRAINSPEECH STUDIO

Este pacote contém as correções para resolver o problema de "uploading infinito" e travamento do treino no GrainSpeech Studio.

---

## Opção 1: Aplicação Automática no Windows (Recomendada)

1. **Feche o GrainSpeech Studio** se estiver rodando (feche a janela do terminal / Iniciar.bat).
2. Abra o **PowerShell** na pasta onde você descompactou este pacote (`GrainSpeech_FIX`).
3. Execute o script de instalação (informando o caminho da pasta raiz do seu Studio):

   ```powershell
   .\APLICAR_FIX.ps1 -Destino "C:\Caminho\Para\GrainSpeechStudio_pacote"
   ```

4. O script fará um **backup automático** dos arquivos originais para `app_backup_<data>` antes de aplicar as correções.

---

## Opção 2: Aplicação Manual

Copie os arquivos da pasta `app/` deste pacote sobrepondo os arquivos correspondentes na pasta `app/` do seu GrainSpeechStudio_pacote:

* `app/gss_app.py`
* `app/finetune_mp3.py`
* `app/bin/gss_prepare.py`
* `app/bin/gss_train.py`
* `app/gss/transcribe.py`

---

## Como Usar Após Aplicar a Correção

1. Inicie o studio normalmente clicando em `Iniciar.bat` ou `PORTABLE-INICIAR.bat`.
2. Acesse a interface web em **`http://127.0.0.1:8756/`**.
3. Na seção **Upload MP3 + Treino Rápido**:
   - Selecione o seu arquivo `.mp3`.
   - Escolha o nome do treino e o número de épocas.
   - Clique em **Upload - Treinar**.
4. O envio do arquivo será concluído via `FormData` e o log exibirá o progresso em tempo real.
5. Quando o treino for concluído, o checkpoint (`.ckpt`) aparecerá automaticamente no menu suspenso **Checkpoint** da seção de Geração de Áudio.
