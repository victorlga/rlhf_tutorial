"""
Script Manim para Animação Visual do Pipeline RLHF
Execute com: manim -pql rlhf_animation.py RLHFPipeline
"""

from manim import *

class RLHFPipeline(Scene):
    """Animação completa e descritiva do pipeline RLHF"""
    
    def construct(self):
        # Título inicial com animação elaborada
        title = Text("RLHF", font_size=72, color=BLUE, weight=BOLD)
        subtitle = Text("Reinforcement Learning from Human Feedback", font_size=28, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.3)
        
        # Underline animado
        underline = Line(LEFT * 3, RIGHT * 3, color=BLUE)
        underline.next_to(subtitle, DOWN, buff=0.2)
        
        self.play(
            Write(title, run_time=1.5),
            FadeIn(subtitle, shift=UP),
        )
        self.play(Create(underline))
        self.wait(1)
        self.play(
            FadeOut(title, shift=UP),
            FadeOut(subtitle, shift=UP),
            FadeOut(underline)
        )
        
        # Mostrar arquitetura geral primeiro
        self.show_architecture_overview()
        
        # Fase 1: Modelo Original
        self.phase_1_original_model()
        
        # Fase 2: Coleta de Feedback
        self.phase_2_feedback_collection()
        
        # Fase 3: Treinamento Reward Model
        self.phase_3_reward_training()
        
        # Fase 4: Treinamento da Policy
        self.phase_4_policy_training()
        
        # Fase 5: Comparação Final
        self.phase_5_comparison()
        
        # Final
        self.final_message()
    
    def show_architecture_overview(self):
        """Mostra visão geral da arquitetura antes de começar"""
        title = Text("Arquitetura do Sistema", font_size=40, color=BLUE)
        title.to_edge(UP)
        
        # Componentes principais
        gpt2_box = self.create_box("GPT-2\nPolicy Model", GREEN, 2.5, 1.8)
        gpt2_box.shift(LEFT * 4.5 + UP * 0.5)
        
        ref_box = self.create_box("Reference\nModel", GRAY, 2, 1.5)
        ref_box.shift(LEFT * 4.5 + DOWN * 2)
        
        reward_box = self.create_box("BERT\nReward Model", PURPLE, 2.5, 1.8)
        reward_box.shift(RIGHT * 4.5)
        
        human_icon = Text("👤", font_size=60)
        human_label = Text("Human\nFeedback", font_size=18, color=BLUE)
        human_label.next_to(human_icon, DOWN, buff=0.2)
        human_group = VGroup(human_icon, human_label)
        human_group.shift(DOWN * 2.5)
        
        # Desenhar componentes
        self.play(Write(title))
        self.play(
            FadeIn(gpt2_box, shift=RIGHT),
            FadeIn(ref_box, shift=RIGHT),
            run_time=1
        )
        self.play(FadeIn(reward_box, shift=LEFT), run_time=1)
        self.play(FadeIn(human_group, scale=0.5), run_time=0.8)
        
        # Conexões com labels
        connections = [
            (gpt2_box, reward_box, "Gera respostas", GREEN, 0.3),
            (reward_box, human_group, "Avalia", PURPLE, 0),
            (human_group, reward_box, "Treina", BLUE, 0),
            (reward_box, gpt2_box, "Otimiza", ORANGE, -0.3),
            (ref_box, gpt2_box, "KL Penalty", RED, 0),
        ]
        
        arrows = []
        for start, end, label_text, color, offset in connections:
            start_point = start.get_edge_center(RIGHT if start.get_x() < end.get_x() else DOWN if start.get_y() > end.get_y() else UP)
            end_point = end.get_edge_center(LEFT if start.get_x() < end.get_x() else UP if start.get_y() > end.get_y() else DOWN)
            
            arrow = CurvedArrow(
                start_point, end_point,
                color=color, angle=offset
            )
            label = Text(label_text, font_size=14, color=color)
            label.next_to(arrow, UP if offset >= 0 else DOWN, buff=0.1)
            
            self.play(Create(arrow), Write(label), run_time=0.6)
            arrows.append(VGroup(arrow, label))
        
        self.wait(2)
        
        # Destacar fluxo sequencial
        highlight = SurroundingRectangle(gpt2_box, color=YELLOW, buff=0.1)
        self.play(Create(highlight))
        
        flow_order = [reward_box, human_group, reward_box, gpt2_box]
        for box in flow_order:
            self.play(
                Transform(highlight, SurroundingRectangle(box, color=YELLOW, buff=0.1)),
                run_time=0.5
            )
        
        self.wait(1)
        self.play(
            FadeOut(highlight),
            *[FadeOut(arrow) for arrow in arrows],
            FadeOut(title),
            FadeOut(gpt2_box),
            FadeOut(ref_box),
            FadeOut(reward_box),
            FadeOut(human_group)
        )
    
    def phase_1_original_model(self):
        """Fase 1: Modelo GPT-2 original com visualização detalhada"""
        # Título com número
        phase_num = Text("1", font_size=60, color=YELLOW, weight=BOLD)
        phase_num.to_corner(UL, buff=0.5)
        
        title = Text("Modelo Original (GPT-2)", font_size=36, color=YELLOW)
        title.next_to(phase_num, RIGHT, buff=0.5)
        
        self.play(Write(phase_num), Write(title))
        
        # Prompt no topo
        prompt_box = RoundedRectangle(
            width=5, height=0.8,
            corner_radius=0.1,
            color=BLUE,
            fill_opacity=0.2
        )
        prompt_text = Text('Prompt: "Donald Trump is a"', font_size=22, color=BLUE)
        prompt_text.move_to(prompt_box)
        prompt_group = VGroup(prompt_box, prompt_text)
        prompt_group.shift(UP * 2.2)
        
        # Animação de entrada do prompt
        self.play(FadeIn(prompt_group, shift=DOWN))
        self.wait(0.3)
        
        # Arrow do prompt para o modelo
        arrow_in = Arrow(
            prompt_group.get_bottom(),
            prompt_group.get_bottom() + DOWN * 0.8,
            buff=0.1,
            color=BLUE,
            stroke_width=6
        )
        arrow_label_in = Text("Input", font_size=14, color=BLUE)
        arrow_label_in.next_to(arrow_in, RIGHT, buff=0.1)
        
        self.play(GrowArrow(arrow_in), Write(arrow_label_in))
        
        # Criar modelo GPT-2 com detalhes
        gpt2_box = self.create_box("GPT-2", GREEN, 4, 2.5)
        gpt2_box.shift(UP * 0.2)
        
        # Adicionar informações técnicas dentro da box
        params = Text("124M parâmetros", font_size=14, color=GRAY)
        params.next_to(gpt2_box[1], DOWN, buff=0.2)
        
        layers = Text("12 Transformer Layers", font_size=12, color=GRAY)
        layers.next_to(params, DOWN, buff=0.1)
        
        model_group = VGroup(gpt2_box, params, layers)
        
        self.play(FadeIn(model_group, scale=0.8))
        self.wait(0.3)
        
        # Simular processamento
        processing = Text("Processando...", font_size=16, color=YELLOW)
        processing.next_to(gpt2_box, RIGHT, buff=0.5)
        self.play(Write(processing))
        
        # Dots animados
        dots = Text("...", font_size=20, color=YELLOW)
        dots.next_to(processing, RIGHT, buff=0.1)
        for _ in range(3):
            self.play(FadeIn(dots), run_time=0.25)
            self.play(FadeOut(dots), run_time=0.25)
        
        self.play(FadeOut(processing))
        
        # Arrow do modelo para resposta
        arrow_out = Arrow(
            gpt2_box.get_bottom(),
            gpt2_box.get_bottom() + DOWN * 0.8,
            buff=0.1,
            color=GREEN,
            stroke_width=6
        )
        arrow_label_out = Text("Output", font_size=14, color=GREEN)
        arrow_label_out.next_to(arrow_out, RIGHT, buff=0.1)
        
        self.play(GrowArrow(arrow_out), Write(arrow_label_out))
        
        # Resposta aparece
        response_box = RoundedRectangle(
            width=6, height=1,
            corner_radius=0.1,
            color=WHITE,
            fill_opacity=0.1,
            stroke_width=3
        )
        response_text = Text(
            'Response: "businessman and politician"',
            font_size=20,
            color=WHITE
        )
        response_text.move_to(response_box)
        response_group = VGroup(response_box, response_text)
        response_group.next_to(arrow_out, DOWN, buff=0.3)
        
        self.play(
            FadeIn(response_group, shift=UP),
            response_box.animate.set_stroke(GREEN, width=3),
            response_box.animate.set_fill(GREEN, opacity=0.1)
        )
        
        # Destaque na resposta com pulse
        highlight = SurroundingRectangle(response_group, color=YELLOW, buff=0.1, stroke_width=4)
        self.play(Create(highlight))
        self.play(
            highlight.animate.set_stroke(width=6),
            run_time=0.3
        )
        self.play(
            highlight.animate.set_stroke(width=4),
            run_time=0.3
        )
        self.play(FadeOut(highlight))
        
        # Mostrar fluxo completo com setas
        flow_highlight = VGroup(
            SurroundingRectangle(prompt_group, color=BLUE, buff=0.15),
            SurroundingRectangle(gpt2_box, color=GREEN, buff=0.15),
            SurroundingRectangle(response_group, color=GREEN, buff=0.15),
        )
        
        self.play(Create(flow_highlight[0]), run_time=0.4)
        self.play(Transform(flow_highlight[0], flow_highlight[1]), run_time=0.4)
        self.play(Transform(flow_highlight[0], flow_highlight[2]), run_time=0.4)
        self.play(FadeOut(flow_highlight[0]))
        
        self.wait(2)
        self.play(
            FadeOut(phase_num),
            FadeOut(title),
            FadeOut(prompt_group),
            FadeOut(arrow_in),
            FadeOut(arrow_label_in),
            FadeOut(model_group),
            FadeOut(arrow_out),
            FadeOut(arrow_label_out),
            FadeOut(response_group)
        )
    
    def phase_2_feedback_collection(self):
        """Fase 2: Coleta de feedback com interação visual"""
        phase_num = Text("2", font_size=60, color=YELLOW, weight=BOLD)
        phase_num.to_corner(UL, buff=0.5)
        
        title = Text("Coleta de Feedback Humano", font_size=36, color=YELLOW)
        title.next_to(phase_num, RIGHT, buff=0.5)
        
        self.play(Write(phase_num), Write(title))
        
        # Modelo à esquerda
        model = self.create_box("GPT-2\nGerando", GREEN, 2.5, 2)
        model.shift(LEFT * 4.5 + UP * 0.5)
        
        self.play(FadeIn(model, shift=RIGHT))
        
        # Gerar múltiplas respostas
        responses_data = [
            ("businessman and\npolitician", WHITE),
            ("controversial\nfigure", WHITE),
            ("former president", WHITE),
        ]
        
        responses = VGroup()
        for i, (text, color) in enumerate(responses_data):
            box = RoundedRectangle(
                width=2.5, height=1,
                corner_radius=0.1,
                color=color,
                fill_opacity=0.1
            )
            label = Text(text, font_size=14, color=color)
            label.move_to(box)
            resp_group = VGroup(box, label)
            resp_group.shift(RIGHT * 0.5 + UP * (1.5 - i * 1.5))
            responses.add(resp_group)
        
        # Animar geração sequencial
        for resp in responses:
            arrow = Arrow(model.get_right(), resp.get_left(), buff=0.1, color=GREEN)
            self.play(
                Create(arrow),
                FadeIn(resp, shift=LEFT),
                run_time=0.6
            )
            self.play(FadeOut(arrow), run_time=0.2)
        
        self.wait(0.5)
        
        # Humano avaliador
        human_circle = Circle(radius=0.8, color=BLUE, fill_opacity=0.3)
        human_icon = Text("👤", font_size=50)
        human_icon.move_to(human_circle)
        human_label = Text("Avaliador", font_size=16, color=BLUE)
        human_label.next_to(human_circle, DOWN, buff=0.2)
        human_group = VGroup(human_circle, human_icon, human_label)
        human_group.shift(RIGHT * 4.5)
        
        self.play(FadeIn(human_group, scale=0.5))
        self.wait(0.5)
        
        # Avaliar cada resposta
        ratings = ["👍", "👎", "😐"]
        colors = [GREEN, RED, YELLOW]
        
        for resp, rating, color in zip(responses, ratings, colors):
            # Arrow de avaliação
            eval_arrow = Arrow(
                resp.get_right(),
                human_group.get_left(),
                buff=0.1,
                color=BLUE
            )
            self.play(Create(eval_arrow), run_time=0.4)
            
            # Pensar
            thinking = Text("...", font_size=24, color=YELLOW)
            thinking.next_to(human_icon, RIGHT, buff=0.1)
            self.play(FadeIn(thinking), run_time=0.3)
            self.play(FadeOut(thinking), run_time=0.3)
            
            # Rating aparece
            rating_icon = Text(rating, font_size=30, color=color)
            rating_icon.next_to(resp, RIGHT, buff=0.3)
            
            self.play(
                FadeIn(rating_icon, scale=0.5),
                resp[0].animate.set_stroke(color=color, width=3),
                run_time=0.4
            )
            
            # Guardar feedback
            feedback_dot = Dot(color=color, radius=0.1)
            feedback_dot.next_to(human_group, DOWN, buff=0.5 + len(self.mobjects) * 0.1)
            self.play(
                Transform(rating_icon, feedback_dot),
                FadeOut(eval_arrow),
                run_time=0.3
            )
        
        # Mostrar dataset acumulado
        dataset_box = RoundedRectangle(
            width=2, height=1.5,
            corner_radius=0.1,
            color=ORANGE,
            fill_opacity=0.2
        )
        dataset_label = Text("Feedback\nDataset", font_size=14, color=ORANGE)
        dataset_label.move_to(dataset_box)
        dataset = VGroup(dataset_box, dataset_label)
        dataset.shift(RIGHT * 4.5 + DOWN * 2.5)
        
        self.play(
            FadeIn(dataset, shift=UP)
        )
        
        self.wait(2)
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )
    
    def phase_3_reward_training(self):
        """Fase 3: Treinamento do reward model com visualização de loss"""
        phase_num = Text("3", font_size=60, color=YELLOW, weight=BOLD)
        phase_num.to_corner(UL, buff=0.5)
        
        title = Text("Treinar Reward Model", font_size=36, color=YELLOW)
        title.next_to(phase_num, RIGHT, buff=0.5)
        
        self.play(Write(phase_num), Write(title))
        
        # Dataset e modelo
        dataset = self.create_box("Feedback\nDataset\n(10 exemplos)", ORANGE, 3, 2)
        dataset.shift(LEFT * 4)
        
        bert = self.create_box("BERT\nReward Model", PURPLE, 3, 2)
        bert.shift(RIGHT * 4)
        
        self.play(
            FadeIn(dataset, shift=RIGHT),
            FadeIn(bert, shift=LEFT)
        )
        
        # Processo de treinamento
        train_arrow = Arrow(
            dataset.get_right(),
            bert.get_left(),
            buff=0.1,
            color=BLUE,
            stroke_width=8
        )
        train_label = Text("Treinamento\nSupervisionado", font_size=14, color=BLUE)
        train_label.next_to(train_arrow, UP, buff=0.1)
        
        self.play(Create(train_arrow), Write(train_label))
        
        # Loss function
        loss_box = RoundedRectangle(
            width=6, height=1.2,
            corner_radius=0.1,
            color=RED,
            fill_opacity=0.1
        )
        loss_formula = MathTex(
            r"\mathcal{L} = \text{MSE}(r_{pred}, r_{human})",
            font_size=32,
            color=WHITE
        )
        loss_formula.move_to(loss_box)
        loss_group = VGroup(loss_box, loss_formula)
        loss_group.shift(DOWN * 2)
        
        self.play(FadeIn(loss_group, shift=UP))
        
        # Explicação
        explanation = Text(
            "Aprende a prever preferências humanas",
            font_size=16,
            color=GRAY
        )
        explanation.next_to(loss_group, DOWN, buff=0.3)
        self.play(Write(explanation))
        
        self.wait(1)
        
        # Animação de convergência com gráfico
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 0.6, 0.2],
            x_length=5,
            y_length=2.5,
            axis_config={"color": GRAY},
        ).shift(RIGHT * 1 + UP * 1.5).scale(0.8)
        
        x_label = Text("Epoch", font_size=12, color=GRAY)
        x_label.next_to(axes.x_axis, DOWN, buff=0.1)
        
        y_label = Text("Loss", font_size=12, color=GRAY)
        y_label.next_to(axes.y_axis, LEFT, buff=0.1)
        
        self.play(
            FadeOut(dataset),
            FadeOut(bert),
            FadeOut(train_arrow),
            FadeOut(train_label),
            FadeOut(loss_group),
            FadeOut(explanation),
            Create(axes),
            Write(x_label),
            Write(y_label)
        )
        
        # Curva de loss decrescente
        loss_values = [0.5, 0.38, 0.28, 0.19, 0.13, 0.09, 0.08, 0.075, 0.072, 0.07]
        points = [axes.c2p(i, val) for i, val in enumerate(loss_values)]
        
        loss_curve = VMobject(color=RED)
        loss_curve.set_points_smoothly(points)
        
        # Animar curva com dots
        dots = VGroup(*[Dot(point, color=YELLOW, radius=0.05) for point in points])
        
        self.play(Create(loss_curve), run_time=3)
        self.play(FadeIn(dots, lag_ratio=0.1), run_time=2)
        
        # Loss final
        final_loss = Text("Loss Final: 0.07", font_size=24, color=GREEN)
        final_loss.next_to(axes, DOWN, buff=0.5)
        
        checkmark = Text("✓", font_size=40, color=GREEN)
        checkmark.next_to(final_loss, LEFT, buff=0.3)
        
        self.play(Write(final_loss), FadeIn(checkmark, scale=0.5))
        
        # Status
        status = Text("Modelo pronto para avaliar respostas!", font_size=20, color=GREEN)
        status.next_to(final_loss, DOWN, buff=0.5)
        self.play(Write(status))
        
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
    
    def phase_4_policy_training(self):
        """Fase 4: Treinamento da policy com PPO - VERSÃO SIMPLIFICADA"""
        # Título
        phase_num = Text("4", font_size=60, color=YELLOW, weight=BOLD)
        phase_num.to_corner(UL, buff=0.5)
        
        title = Text("Otimização da Policy (PPO)", font_size=36, color=YELLOW)
        title.next_to(phase_num, RIGHT, buff=0.5)
        
        self.play(Write(phase_num))
        self.play(Write(title))
        self.wait(1)
        
        # Layout: 3 caixas em linha horizontal
        # Reference Model (esquerda)
        ref_box = RoundedRectangle(
            width=2.8, height=2,
            corner_radius=0.2,
            color=GRAY,
            fill_opacity=0.3,
            stroke_width=3
        )
        ref_label = Text("Reference\nModel\n(frozen)", font_size=16, color=GRAY)
        ref_label.move_to(ref_box)
        ref_model = VGroup(ref_box, ref_label)
        ref_model.shift(LEFT * 4.5 + DOWN * 0.5)
        
        # Policy Model (centro-esquerda)
        policy_box = RoundedRectangle(
            width=2.8, height=2,
            corner_radius=0.2,
            color=GREEN,
            fill_opacity=0.3,
            stroke_width=3
        )
        policy_label = Text("Policy\nModel\n(training)", font_size=16, color=GREEN)
        policy_label.move_to(policy_box)
        policy_model = VGroup(policy_box, policy_label)
        policy_model.shift(LEFT * 1 + DOWN * 0.5)
        
        # Reward Model (direita)
        reward_box = RoundedRectangle(
            width=2.8, height=2,
            corner_radius=0.2,
            color=PURPLE,
            fill_opacity=0.3,
            stroke_width=3
        )
        reward_label = Text("Reward\nModel", font_size=16, color=PURPLE)
        reward_label.move_to(reward_box)
        reward_model = VGroup(reward_box, reward_label)
        reward_model.shift(RIGHT * 4.5 + DOWN * 0.5)
        
        # Mostrar os três modelos
        self.play(FadeIn(ref_model))
        self.wait(0.3)
        self.play(FadeIn(policy_model))
        self.wait(0.3)
        self.play(FadeIn(reward_model))
        self.wait(0.5)
        
        # Arrows conectando os modelos
        # Policy -> Reward (geração)
        arrow1 = Arrow(
            policy_model.get_right(),
            reward_model.get_left(),
            buff=0.1,
            color=GREEN,
            stroke_width=4
        )
        label1 = Text("Generate", font_size=14, color=GREEN)
        label1.next_to(arrow1, UP, buff=0.1)
        
        self.play(GrowArrow(arrow1))
        self.play(Write(label1))
        self.wait(0.5)
        
        # Reward -> Policy (reward signal)
        arrow2 = Arrow(
            reward_model.get_left(),
            policy_model.get_right(),
            buff=0.1,
            color=PURPLE,
            stroke_width=4
        )
        label2 = Text("Reward", font_size=14, color=PURPLE)
        label2.next_to(arrow2, DOWN, buff=0.1)
        
        self.play(GrowArrow(arrow2))
        self.play(Write(label2))
        self.wait(0.5)
        
        # Reference -> Policy (KL penalty)
        arrow3 = Arrow(
            ref_model.get_right(),
            policy_model.get_left(),
            buff=0.1,
            color=RED,
            stroke_width=4
        )
        label3 = Text("KL Penalty", font_size=14, color=RED)
        label3.next_to(arrow3, UP, buff=0.1)
        
        self.play(GrowArrow(arrow3))
        self.play(Write(label3))
        self.wait(1)
        
        # Loss function
        loss_text = MathTex(
            r"Loss = -reward + \beta \cdot KL",
            font_size=28,
            color=WHITE
        )
        loss_text.to_edge(UP, buff=1.5)
        
        self.play(Write(loss_text))
        self.wait(1)
        
        # Mover tudo para cima para dar espaço à barra
        group_to_move = VGroup(
            ref_model, policy_model, reward_model,
            arrow1, label1, arrow2, label2, arrow3, label3
        )
        
        self.play(
            group_to_move.animate.shift(UP * 0.8),
            loss_text.animate.shift(UP * 0.5),
            run_time=0.8
        )
        
        # Progress Bar
        bar_title = Text("Treinamento:", font_size=20, color=YELLOW)
        bar_title.to_edge(DOWN, buff=2.5)
        
        bar_bg = Rectangle(
            width=10,
            height=0.6,
            color=GRAY,
            fill_opacity=0.2,
            stroke_width=2
        )
        bar_bg.next_to(bar_title, DOWN, buff=0.3)
        
        self.play(Write(bar_title))
        self.play(Create(bar_bg))
        
        # Step counter
        step_counter = Text("Step 0/20", font_size=18, color=WHITE)
        step_counter.next_to(bar_bg, DOWN, buff=0.3)
        self.play(Write(step_counter))
        
        # Animação de preenchimento da barra
        for i in range(1, 21):
            # Barra de progresso
            bar_fill = Rectangle(
                width=10 * (i / 20),
                height=0.6,
                color=GREEN,
                fill_opacity=0.8,
                stroke_width=0
            )
            bar_fill.move_to(bar_bg)
            bar_fill.align_to(bar_bg, LEFT)
            
            # Atualizar contador
            new_counter = Text(f"Step {i}/20", font_size=18, color=WHITE)
            new_counter.move_to(step_counter)
            
            # Animações
            if i == 1:
                self.play(
                    Create(bar_fill),
                    Transform(step_counter, new_counter),
                    run_time=0.15
                )
            else:
                self.play(
                    bar_fill.animate.set_width(10 * (i / 20)),
                    Transform(step_counter, new_counter),
                    run_time=0.1
                )
            
            # Pulse no policy a cada 5 steps
            if i % 5 == 0:
                self.play(
                    policy_box.animate.set_fill(GREEN, opacity=0.6),
                    run_time=0.1
                )
                self.play(
                    policy_box.animate.set_fill(GREEN, opacity=0.3),
                    run_time=0.1
                )
        
        # Mensagem final
        check = Text("✓", font_size=50, color=GREEN)
        complete_msg = Text("Completo!", font_size=24, color=GREEN)
        final_group = VGroup(check, complete_msg).arrange(RIGHT, buff=0.3)
        final_group.next_to(step_counter, DOWN, buff=0.5)
        
        self.play(FadeIn(check, scale=0.5))
        self.play(Write(complete_msg))
        self.wait(2)
        
        # Limpar tudo
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1
        )
    
    def phase_5_comparison(self):
        """Fase 5: Comparação visual antes/depois"""
        phase_num = Text("5", font_size=60, color=YELLOW, weight=BOLD)
        phase_num.to_corner(UL, buff=0.5)
        
        title = Text("Comparação: Antes vs Depois", font_size=36, color=YELLOW)
        title.next_to(phase_num, RIGHT, buff=0.5)
        
        self.play(Write(phase_num), Write(title))
        
        # Divisória central
        divider = Line(UP * 3, DOWN * 3, color=GRAY)
        self.play(Create(divider))
        
        # Lado esquerdo - Original
        original_title = Text("Modelo Original", font_size=24, color=GRAY)
        original_title.shift(LEFT * 3.5 + UP * 2.5)
        
        original_box = RoundedRectangle(
            width=3.5, height=4.5,
            corner_radius=0.2,
            color=GRAY,
            fill_opacity=0.1
        )
        original_box.shift(LEFT * 3.5 + DOWN * 0.3)
        
        original_responses = VGroup(
            Text("• controversial figure", font_size=14),
            Text("• billionaire mogul", font_size=14),
            Text("• divisive politician", font_size=14),
            Text("• outspoken businessman", font_size=14),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        original_responses.move_to(original_box)
        
        original_score = Text("Avg Reward: 0.45", font_size=18, color=YELLOW)
        original_score.next_to(original_box, DOWN, buff=0.3)
        
        # Lado direito - RLHF
        rlhf_title = Text("Modelo RLHF", font_size=24, color=GREEN)
        rlhf_title.shift(RIGHT * 3.5 + UP * 2.5)
        
        rlhf_box = RoundedRectangle(
            width=3.5, height=4.5,
            corner_radius=0.2,
            color=GREEN,
            fill_opacity=0.1
        )
        rlhf_box.shift(RIGHT * 3.5 + DOWN * 0.3)
        
        rlhf_responses = VGroup(
            Text("• businessman from NY", font_size=14),
            Text("• 45th president of US", font_size=14),
            Text("• entrepreneur and...", font_size=14),
            Text("• real estate developer", font_size=14),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        rlhf_responses.move_to(rlhf_box)
        
        rlhf_score = Text("Avg Reward: 0.78", font_size=18, color=GREEN)
        rlhf_score.next_to(rlhf_box, DOWN, buff=0.3)
        
        # Animar entrada do lado esquerdo
        self.play(
            Write(original_title),
            Create(original_box),
            run_time=1
        )
        
        for response in original_responses:
            self.play(FadeIn(response, shift=UP), run_time=0.4)
        
        self.play(Write(original_score))
        
        # Animar entrada do lado direito
        self.play(
            Write(rlhf_title),
            Create(rlhf_box),
            run_time=1
        )
        
        for response in rlhf_responses:
            self.play(FadeIn(response, shift=UP), run_time=0.4)
        
        self.play(Write(rlhf_score))
        
        self.wait(1)
        
        # Destacar diferença de scores
        score_diff = Arrow(
            original_score.get_right(),
            rlhf_score.get_left(),
            buff=0.2,
            color=GREEN,
            stroke_width=6
        )
        improvement = Text("+73%", font_size=24, color=GREEN, weight=BOLD)
        improvement.next_to(score_diff, UP, buff=0.1)
        
        self.play(
            GrowArrow(score_diff),
            Write(improvement)
        )
        
        # Análise visual das diferenças
        analysis_title = Text("Análise das Mudanças:", font_size=20, color=BLUE)
        analysis_title.to_edge(DOWN, buff=2)
        
        changes = VGroup(
            Text("✓ Linguagem mais neutra", font_size=14, color=GREEN),
            Text("✓ Foco em fatos objetivos", font_size=14, color=GREEN),
            Text("✓ Menos adjetivos carregados", font_size=14, color=GREEN),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        changes.next_to(analysis_title, DOWN, buff=0.3)
        
        self.play(Write(analysis_title))
        for change in changes:
            self.play(FadeIn(change, shift=RIGHT), run_time=0.5)
        
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
    
    def final_message(self):
        """Mensagem final com sumário"""
        # Título final
        title = Text("RLHF: Resumo", font_size=48, color=BLUE, weight=BOLD)
        title.shift(UP * 2.5)
        
        self.play(Write(title))
        
        # Passos do pipeline
        steps = VGroup(
            self.create_step("1", "Modelo original gera respostas", GREEN),
            self.create_step("2", "Humano avalia qualidade", BLUE),
            self.create_step("3", "Reward model aprende preferências", PURPLE),
            self.create_step("4", "Policy otimiza com PPO", ORANGE),
            self.create_step("5", "Modelo alinhado com humanos", GREEN),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        steps.shift(DOWN * 0.5)
        
        for step in steps:
            self.play(FadeIn(step, shift=RIGHT), run_time=0.5)
        
        self.wait(1)
        
        # Fórmula-chave
        key_formula = MathTex(
            r"\mathcal{L} = -\mathbb{E}[r_\theta] + \beta \cdot D_{KL}(\pi || \pi_{ref})",
            font_size=28,
            color=YELLOW
        )
        key_formula.shift(DOWN * 2.8)
        
        formula_label = Text("Fórmula-chave do RLHF", font_size=16, color=GRAY)
        formula_label.next_to(key_formula, DOWN, buff=0.2)
        
        self.play(
            Write(key_formula),
            FadeIn(formula_label)
        )
        
        self.wait(2)
        
        # Final fadeout
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        # Mensagem de encerramento
        final_text = Text(
            "Alinhamento de modelos de linguagem\ncom valores humanos",
            font_size=32,
            color=BLUE
        )
        
        checkmark = Text("✓", font_size=80, color=GREEN)
        checkmark.next_to(final_text, UP, buff=0.5)
        
        self.play(
            FadeIn(checkmark, scale=0.5),
            Write(final_text)
        )
        
        self.wait(3)
        self.play(FadeOut(checkmark), FadeOut(final_text))
    
    # Métodos auxiliares
    def create_box(self, text, color, width, height):
        """Cria uma caixa com label"""
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.2,
            color=color,
            fill_opacity=0.3,
            stroke_width=3
        )
        label = Text(text, font_size=18, color=color)
        label.move_to(box)
        return VGroup(box, label)
    
    def create_step(self, number, text, color):
        """Cria um item de step numerado"""
        num_circle = Circle(radius=0.3, color=color, fill_opacity=0.5)
        num_text = Text(number, font_size=20, color=WHITE, weight=BOLD)
        num_text.move_to(num_circle)
        num_group = VGroup(num_circle, num_text)
        
        step_text = Text(text, font_size=18, color=WHITE)
        step_text.next_to(num_group, RIGHT, buff=0.3)
        
        return VGroup(num_group, step_text)


class RLHFQuickDemo(Scene):
    """Versão rápida para demonstração (30 segundos)"""
    
    def construct(self):
        # Título
        title = Text("RLHF em 30 segundos", font_size=48, color=BLUE)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))
        
        # Pipeline visual compacto
        gpt2 = self.create_component("GPT-2", GREEN)
        gpt2.shift(LEFT * 5 + UP)
        
        human = self.create_component("👤\nHuman", BLUE)
        human.shift(UP)
        
        reward = self.create_component("Reward", PURPLE)
        reward.shift(RIGHT * 5 + UP)
        
        # Animar fluxo
        self.play(
            FadeIn(gpt2, shift=RIGHT),
            FadeIn(human, scale=0.5),
            FadeIn(reward, shift=LEFT),
            run_time=1
        )
        
        # Arrows
        arrow1 = Arrow(gpt2, human, color=GREEN)
        arrow2 = Arrow(human, reward, color=BLUE)
        arrow3 = CurvedArrow(reward.get_bottom(), gpt2.get_bottom(), angle=-0.5, color=ORANGE)
        
        self.play(Create(arrow1))
        self.play(Create(arrow2))
        self.play(Create(arrow3))
        
        # Labels
        labels = VGroup(
            Text("Gera", font_size=14, color=GREEN).next_to(arrow1, UP, buff=0.05),
            Text("Avalia", font_size=14, color=BLUE).next_to(arrow2, UP, buff=0.05),
            Text("Otimiza", font_size=14, color=ORANGE).next_to(arrow3, DOWN, buff=0.05),
        )
        
        self.play(Write(labels))
        
        # Resultado
        result = Text("= Modelo Alinhado ✓", font_size=32, color=GREEN)
        result.shift(DOWN * 2)
        self.play(Write(result))
        
        self.wait(2)
    
    def create_component(self, text, color):
        """Cria componente simplificado"""
        circle = Circle(radius=0.8, color=color, fill_opacity=0.3, stroke_width=3)
        label = Text(text, font_size=16, color=color)
        label.move_to(circle)
        return VGroup(circle, label)


# Instruções de uso
"""
INSTRUÇÕES DE USO:

1. Instalar Manim:
   pip install manim

2. Renderizar animação completa (recomendado):
   manim -pqh rlhf_animation.py RLHFPipeline

3. Renderizar preview rápido (baixa qualidade):
   manim -pql rlhf_animation.py RLHFPipeline

4. Renderizar versão curta (30s):
   manim -pqh rlhf_animation.py RLHFQuickDemo

5. Renderizar apenas uma fase específica (para debug):
   # Edite o método construct() para chamar apenas a fase desejada
   # Exemplo: self.phase_3_reward_training()

PARÂMETROS:
-p : Preview (abre vídeo automaticamente)
-q : Qualidade (l=low, m=medium, h=high, k=4k)
-s : Renderizar última cena apenas
-a : Renderizar todas as cenas

OUTPUTS:
- Vídeos salvos em: media/videos/rlhf_animation/
- Resolução HD: 1920x1080 (com -qh)
- Resolução 4K: 3840x2160 (com -qk)

DURAÇÃO ESTIMADA:
- RLHFPipeline: ~2-3 minutos
- RLHFQuickDemo: ~30 segundos

CORES USADAS:
- GREEN: Policy/GPT-2
- PURPLE: Reward Model
- BLUE: Human/Feedback
- ORANGE: Training/Optimization
- RED: KL Penalty
- GRAY: Reference Model
- YELLOW: Highlights/Titles
"""