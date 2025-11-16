# ========================================================
# Seção 1: Importações de Bibliotecas
# ========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BertTokenizer,
    BertModel
)
import numpy as np
from tqdm.auto import tqdm
import warnings
import copy
warnings.filterwarnings('ignore')

# ========================================================
# Seção 2: Configuração do Dispositivo
# ========================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Usando MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Usando CUDA")
else:
    device = torch.device("cpu")
    print("⚠️ Usando CPU")

print(f"Device: {device}")

# ========================================================
# Seção 3: Definição do Modelo de Recompensa (BERTRewardModel)
# ========================================================
class BERTRewardModel(nn.Module):
    """
    Modelo de recompensa baseado em BERT que aprende com feedback humano.
    Prediz um score de qualidade para cada resposta gerada.
    """
    def __init__(self, model_name='bert-base-uncased', freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Camada de recompensa: prediz score de 0 (ruim) a 1 (bom)
        self.reward_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output entre 0 e 1
        )
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        reward_score = self.reward_head(cls_output)
        return reward_score

# ========================================================
# Seção 4: Carregamento de Modelos e Tokenizers
# ========================================================
# Carregar modelo generativo e tokenizer (Inglês)
gen_model_name = 'gpt2'
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_tokenizer.pad_token = gen_tokenizer.eos_token
generation_model = AutoModelForCausalLM.from_pretrained(gen_model_name).to(device)
ref_model = copy.deepcopy(generation_model).to(device)

# Carregar tokenizer e modelo para reward (BERT Inglês)
bert_model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Modelo de recompensa
reward_model = BERTRewardModel(model_name=bert_model_name, freeze_bert=False).to(device)

# Otimizadores
reward_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=2e-5, weight_decay=0.01)
policy_optimizer = torch.optim.AdamW(generation_model.parameters(), lr=1e-5, weight_decay=0.01)

print("✅ Modelos inicializados")
print(f"📊 Parâmetros treináveis (reward): {sum(p.numel() for p in reward_model.parameters() if p.requires_grad):,}")
print(f"📊 Parâmetros treináveis (policy): {sum(p.numel() for p in generation_model.parameters() if p.requires_grad):,}")

# ========================================================
# Seção 5: Função para Geração de Completions
# ========================================================
def generate_completion(model, tokenizer, prompt, max_new_tokens=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = full_text[len(prompt):].strip()
    return completion

# ========================================================
# Seção 6: Coletor de Feedback Humano
# ========================================================
class HumanFeedbackCollector:
    def __init__(self):
        self.feedbacks = []
        self.current_response = None
    
    def show_response_and_collect(self, prompt, completion, model_score):
        full_sentence = f"{prompt} {completion}"
        self.current_response = {
            'prompt': prompt,
            'completion': completion,
            'model_score': model_score
        }
        
        print(f"\n💬 Prompt: {prompt}")
        print(f"🤖 Resposta: {full_sentence}")
        print("Avalie a qualidade da resposta. Digite 1 para bom, 2 para neutro, 3 para ruim, ou q para sair.")
        
        while True:
            user_input = input("Avalie (1/2/3/q): ").strip().lower()
            if user_input == '1':
                self._record_feedback(1.0)
                return False
            elif user_input == '2':
                self._record_feedback(0.5)
                return False
            elif user_input == '3':
                self._record_feedback(0.0)
                return False
            elif user_input == 'q':
                return True
            else:
                print("Entrada inválida. Por favor, digite 1, 2, 3 ou q.")
    
    def _record_feedback(self, reward):
        self.current_response['human_reward'] = reward
        self.feedbacks.append(self.current_response.copy())
        if reward == 1.0:
            emoji = "👍"
        elif reward == 0.5:
            emoji = "😐"
        else:
            emoji = "👎"
        print(f"{emoji} Feedback registrado!")
    
    def get_feedback_batch(self):
        return self.feedbacks

feedback_collector = HumanFeedbackCollector()
print("✅ Sistema de feedback inicializado")

# ========================================================
# Seção 7: Função para Coleta e Treinamento do Modelo de Recompensa
# ========================================================
def collect_and_train_reward(model, prompt, gen_model, gen_tokenizer, bert_tokenizer, feedback_collector, optimizer, device):
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"🎯 Coleta de Feedback")
    print(f"{'='*60}\n")
    
    while True:
        completion = generate_completion(gen_model, gen_tokenizer, prompt)
        
        text = f"{prompt} [SEP] {completion}"
        encoding = bert_tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
        
        with torch.no_grad():
            model_score = model(encoding['input_ids'], encoding['attention_mask']).item()
        
        stop = feedback_collector.show_response_and_collect(prompt, completion, model_score)
        if stop:
            break
    
    print(f"\n{'='*60}")
    print(f"🔥 Treinando Reward Model com Feedback")
    print(f"{'='*60}\n")
    
    feedbacks = feedback_collector.get_feedback_batch()
    
    if len(feedbacks) == 0:
        print("⚠️ Nenhum feedback coletado")
        return 0.0
    
    model.train()
    total_loss = 0.0
    
    for fb in tqdm(feedbacks, desc="Atualizando reward model"):
        text = f"{fb['prompt']} [SEP] {fb['completion']}"
        encoding = bert_tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
        
        optimizer.zero_grad()
        predicted_reward = model(encoding['input_ids'], encoding['attention_mask'])
        human_reward = torch.tensor([[fb['human_reward']]], dtype=torch.float32).to(device)
        loss = 2.0 *F.mse_loss(predicted_reward, human_reward)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(feedbacks)
    print(f"\n✅ Loss médio: {avg_loss:.4f}")
    feedback_collector.feedbacks = []
    return avg_loss

# ========================================================
# Seção 8: Treinamento Direto da Policy com Reward Model
# ========================================================
def train_policy_with_reward(gen_model, ref_model, reward_model, tokenizer, bert_tokenizer, 
                             prompt, optimizer, device, num_steps=20, batch_size=4, kl_coef=0.1):
    """
    Treina a policy diretamente usando o reward model.
    Otimiza: reward - kl_penalty
    """
    print(f"\n{'='*60}")
    print("🔥 Treinando Policy com Reward Model")
    print(f"{'='*60}\n")
    
    gen_model.train()
    ref_model.eval()
    reward_model.eval()
    
    for step in tqdm(range(num_steps), desc="Passos de treinamento"):
        optimizer.zero_grad()
        total_loss = 0.0
        
        for _ in range(batch_size):
            # Gerar resposta
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Gerar tokens com gradientes
            outputs = gen_model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            generated_ids = outputs.sequences
            completion_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            completion = completion_text[len(prompt):].strip()
            
            # Calcular reward
            text = f"{prompt} [SEP] {completion}"
            encoding = bert_tokenizer(text, truncation=True, padding='max_length', 
                                     max_length=128, return_tensors='pt').to(device)
            
            with torch.no_grad():
                reward = reward_model(encoding['input_ids'], encoding['attention_mask'])
            
            # Calcular log probs da policy atual e referência
            gen_outputs = gen_model(generated_ids, attention_mask=torch.ones_like(generated_ids))
            with torch.no_grad():
                ref_outputs = ref_model(generated_ids, attention_mask=torch.ones_like(generated_ids))
            
            gen_logits = gen_outputs.logits[:, :-1, :]
            ref_logits = ref_outputs.logits[:, :-1, :]
            target_ids = generated_ids[:, 1:]
            
            gen_log_probs = F.log_softmax(gen_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            gen_log_prob = gen_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).sum()
            ref_log_prob = ref_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).sum()
            
            # KL divergence penalty
            kl_penalty = gen_log_prob - ref_log_prob
            
            # Loss: maximizar reward - kl_penalty
            # Equivalente a minimizar: -reward + kl_coef * kl_penalty
            loss = -reward.squeeze() + kl_coef * kl_penalty
            total_loss += loss
        
        # Backprop
        avg_loss = total_loss / batch_size
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"\nStep {step+1}/{num_steps} - Loss: {avg_loss.item():.4f}")
    
    print("\n✅ Treinamento da policy completo!")

# ========================================================
# Seção 9: Prompt Fixo
# ========================================================
fixed_prompt = "Donald Trump is a"

print("🎬 Iniciando RLHF!\n")

# ========================================================
# Seção 10: Pipeline Principal
# ========================================================
# 1. Coleta e treino do reward
loss = collect_and_train_reward(
    model=reward_model,
    prompt=fixed_prompt,
    gen_model=generation_model,
    gen_tokenizer=gen_tokenizer,
    bert_tokenizer=bert_tokenizer,
    feedback_collector=feedback_collector,
    optimizer=reward_optimizer,
    device=device
)

# 2. Treina policy usando reward model
train_policy_with_reward(
    gen_model=generation_model,
    ref_model=ref_model,
    reward_model=reward_model,
    tokenizer=gen_tokenizer,
    bert_tokenizer=bert_tokenizer,
    prompt=fixed_prompt,
    optimizer=policy_optimizer,
    device=device,
    num_steps=20,
    batch_size=4,
    kl_coef=0.1
)

print("\n🎉 Treinamento Completo!")

# ========================================================
# Seção 11: Função para Geração de 10 Frases
# ========================================================
def generate_10_sentences(model, tokenizer, prompt):
    sentences = []
    for _ in range(10):
        completion = generate_completion(model, tokenizer, prompt)
        full_sentence = f"{prompt} {completion}"
        sentences.append(full_sentence)
    return sentences

# ========================================================
# Seção 12: Geração e Impressão de Frases para Comparação
# ========================================================
print(f"\n{'='*60}")
print("📊 10 Frases sem o novo treino (modelo original)")
print(f"{'='*60}\n")
original_sentences = generate_10_sentences(ref_model, gen_tokenizer, fixed_prompt)
for i, sent in enumerate(original_sentences, 1):
    print(f"{i}. {sent}")

print(f"\n{'='*60}")
print("📊 10 Frases com o novo treino (modelo treinado)")
print(f"{'='*60}\n")
trained_sentences = generate_10_sentences(generation_model, gen_tokenizer, fixed_prompt)
for i, sent in enumerate(trained_sentences, 1):
    print(f"{i}. {sent}")

# ========================================================
# Seção 13: Salvamento dos Modelos
# ========================================================
torch.save({
    'generation_model_state_dict': generation_model.state_dict(),
    'reward_model_state_dict': reward_model.state_dict(),
    'reward_optimizer_state_dict': reward_optimizer.state_dict(),
    'policy_optimizer_state_dict': policy_optimizer.state_dict(),
}, 'rlhf_model.pt')

print("✅ Modelos salvos!")