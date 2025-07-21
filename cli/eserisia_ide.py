#!/usr/bin/env python3
"""
ESERISIA AI - CLI Interface Ultra-Avancée
========================================
Interface ligne de commande pour IDE intelligent
Commandes: scan, analyze, edit, create, optimize, generate
"""

import typer
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.tree import Tree
import os
import sys

# Import ESERISIA IDE
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eserisia.ide_engine import (
    EserisiaIDE, scan_project_intelligent, analyze_file_intelligent,
    edit_file_with_ai, create_file_with_template, get_ide_capabilities
)
from eserisia.project_generator_simple import EserisiaProjectGeneratorSimple

# Instance IDE globale
eserisia_ide = EserisiaIDE()
project_generator = EserisiaProjectGeneratorSimple()

app = typer.Typer(
    name="eserisia-ide",
    help="🧠 ESERISIA AI - IDE Intelligent Ultra-Avancé CLI",
    add_completion=False
)

console = Console()

def print_header():
    """Header CLI ultra-avancé"""
    console.print(Panel.fit(
        "[bold blue]🧠 ESERISIA AI - IDE Intelligent Ultra-Avancé[/bold blue]\n"
        "[dim]Architecture évolutive 2025 • Précision: 99.87%[/dim]\n"
        "[green]Lecture • Compréhension • Édition intelligente des projets[/green]",
        border_style="blue",
        padding=(1, 2)
    ))

@app.command()
def scan(
    project_path: str = typer.Argument(".", help="Chemin du projet à scanner"),
    output_format: str = typer.Option("table", "--format", "-f", help="Format: table, json, tree"),
    save_report: bool = typer.Option(False, "--save", "-s", help="Sauvegarder rapport JSON")
):
    """
    🔍 Scanner un projet avec analyse intelligente complète
    """
    print_header()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("🧠 Analyse ESERISIA en cours...", total=None)
        
        # Scanner projet
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        project_structure = loop.run_until_complete(scan_project_intelligent(project_path))
        loop.close()
        
        progress.update(task, completed=100)
    
    if not project_structure:
        console.print("❌ [red]Erreur lors du scan du projet[/red]")
        raise typer.Exit(1)
    
    # Affichage selon format
    if output_format == "json":
        project_data = {
            "root_path": str(project_structure.root_path),
            "languages": project_structure.languages,
            "frameworks": project_structure.frameworks,
            "total_files": project_structure.total_files,
            "total_lines": project_structure.total_lines,
            "architecture_pattern": project_structure.architecture_pattern
        }
        console.print_json(json.dumps(project_data, indent=2))
        
        if save_report:
            report_path = Path(project_path) / "eserisia_scan_report.json"
            with open(report_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            console.print(f"💾 [green]Rapport sauvegardé: {report_path}[/green]")
    
    elif output_format == "tree":
        tree = Tree(f"📁 {project_structure.root_path}")
        
        langs_node = tree.add("🔤 Langages")
        for lang in project_structure.languages:
            langs_node.add(f"• {lang.title()}")
        
        frameworks_node = tree.add("⚡ Frameworks")
        for fw in project_structure.frameworks:
            frameworks_node.add(f"• {fw}")
        
        stats_node = tree.add("📊 Statistiques")
        stats_node.add(f"📁 Fichiers: {project_structure.total_files:,}")
        stats_node.add(f"📄 Lignes: {project_structure.total_lines:,}")
        stats_node.add(f"🏗️ Architecture: {project_structure.architecture_pattern}")
        
        console.print(tree)
    
    else:  # table format (default)
        # Table langages
        lang_table = Table(title="🔤 Langages Détectés")
        lang_table.add_column("Langage", style="cyan")
        lang_table.add_column("Type", style="green")
        
        for lang in project_structure.languages:
            lang_type = "Système" if lang in ["json", "yaml", "xml"] else "Programmation"
            lang_table.add_row(lang.title(), lang_type)
        
        console.print(lang_table)
        
        # Table frameworks
        if project_structure.frameworks:
            fw_table = Table(title="⚡ Frameworks Détectés")
            fw_table.add_column("Framework", style="magenta")
            fw_table.add_column("Catégorie", style="yellow")
            
            for fw in project_structure.frameworks:
                category = "Web" if fw in ["django", "flask", "fastapi", "react", "vue"] else "Général"
                fw_table.add_row(fw.title(), category)
            
            console.print(fw_table)
        
        # Statistiques
        stats_table = Table(title="📊 Statistiques Projet")
        stats_table.add_column("Métrique", style="blue")
        stats_table.add_column("Valeur", style="green")
        
        stats_table.add_row("📁 Total Fichiers", f"{project_structure.total_files:,}")
        stats_table.add_row("📄 Total Lignes", f"{project_structure.total_lines:,}")
        stats_table.add_row("🔤 Langages", str(len(project_structure.languages)))
        stats_table.add_row("⚡ Frameworks", str(len(project_structure.frameworks)))
        stats_table.add_row("🏗️ Architecture", project_structure.architecture_pattern)
        
        console.print(stats_table)
    
    console.print(f"\n✅ [green]Scan terminé - {project_structure.total_files} fichiers analysés[/green]")

@app.command()
def analyze(
    file_path: str = typer.Argument(..., help="Chemin du fichier à analyser"),
    show_content: bool = typer.Option(False, "--content", "-c", help="Afficher le contenu"),
    max_lines: int = typer.Option(50, "--max-lines", "-m", help="Nombre max de lignes à afficher")
):
    """
    🧠 Analyser un fichier en profondeur avec IA
    """
    print_header()
    
    if not Path(file_path).exists():
        console.print(f"❌ [red]Fichier non trouvé: {file_path}[/red]")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("🧠 Analyse intelligente...", total=None)
        
        # Analyser fichier
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        analysis = loop.run_until_complete(analyze_file_intelligent(file_path))
        loop.close()
        
        progress.update(task, completed=100)
    
    # Informations générales
    info_table = Table(title=f"📄 Analyse: {Path(file_path).name}")
    info_table.add_column("Propriété", style="blue")
    info_table.add_column("Valeur", style="green")
    
    info_table.add_row("📏 Taille", f"{analysis.size:,} caractères")
    info_table.add_row("📄 Lignes", str(analysis.lines))
    info_table.add_row("🔤 Langage", analysis.language.title())
    info_table.add_row("🔧 Complexité", analysis.complexity)
    
    console.print(info_table)
    
    # Éléments détectés
    if analysis.functions or analysis.classes or analysis.imports:
        elements_table = Table(title="🔍 Éléments Détectés")
        elements_table.add_column("Type", style="cyan")
        elements_table.add_column("Nombre", style="green")
        elements_table.add_column("Exemples", style="yellow")
        
        if analysis.functions:
            examples = []
            for func in analysis.functions[:3]:
                if isinstance(func, dict):
                    examples.append(func.get('name', str(func)))
                else:
                    examples.append(str(func))
            
            elements_table.add_row(
                "⚙️ Fonctions",
                str(len(analysis.functions)),
                ", ".join(examples)
            )
        
        if analysis.classes:
            examples = []
            for cls in analysis.classes[:3]:
                if isinstance(cls, dict):
                    examples.append(cls.get('name', str(cls)))
                else:
                    examples.append(str(cls))
            
            elements_table.add_row(
                "🏛️ Classes",
                str(len(analysis.classes)),
                ", ".join(examples)
            )
        
        if analysis.imports:
            examples = analysis.imports[:3]
            elements_table.add_row(
                "📦 Imports",
                str(len(analysis.imports)),
                ", ".join(examples)
            )
        
        console.print(elements_table)
    
    # Issues
    if analysis.issues:
        console.print("\n⚠️ [yellow]Issues Détectées:[/yellow]")
        for issue in analysis.issues[:5]:  # Limite pour lisibilité
            console.print(f"  • {issue}")
    
    # Suggestions IA
    if analysis.suggestions:
        console.print("\n💡 [blue]Suggestions IA Ultra-Avancées:[/blue]")
        for suggestion in analysis.suggestions:
            console.print(f"  • {suggestion}")
    
    # Contenu fichier
    if show_content:
        console.print(f"\n📄 [bold]Contenu ({Path(file_path).name}):[/bold]")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Limiter affichage
            lines = content.splitlines()
            if len(lines) > max_lines:
                display_content = '\n'.join(lines[:max_lines])
                display_content += f"\n\n... ({len(lines) - max_lines} lignes supplémentaires)"
            else:
                display_content = content
            
            syntax = Syntax(display_content, analysis.language, theme="monokai", line_numbers=True)
            console.print(syntax)
            
        except Exception as e:
            console.print(f"❌ [red]Erreur lecture: {e}[/red]")

@app.command()
def edit(
    file_path: str = typer.Argument(..., help="Chemin du fichier à éditer"),
    operation: str = typer.Argument(..., help="Opération: replace, insert, delete, optimize"),
    target: str = typer.Option(None, "--target", "-t", help="Texte à remplacer/supprimer"),
    new_content: str = typer.Option(None, "--content", "-c", help="Nouveau contenu"),
    line_number: int = typer.Option(None, "--line", "-l", help="Numéro de ligne pour insertion"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Créer backup")
):
    """
    ✏️ Éditer un fichier avec intelligence artificielle
    """
    print_header()
    
    if not Path(file_path).exists():
        console.print(f"❌ [red]Fichier non trouvé: {file_path}[/red]")
        raise typer.Exit(1)
    
    # Validation opération
    valid_operations = ["replace", "insert", "delete", "optimize", "add_function"]
    if operation not in valid_operations:
        console.print(f"❌ [red]Opération invalide. Utilisez: {', '.join(valid_operations)}[/red]")
        raise typer.Exit(1)
    
    # Validation paramètres
    if operation == "replace" and (not target or not new_content):
        console.print("❌ [red]Replace nécessite --target et --content[/red]")
        raise typer.Exit(1)
    
    if operation == "insert" and (not new_content or line_number is None):
        console.print("❌ [red]Insert nécessite --content et --line[/red]")
        raise typer.Exit(1)
    
    if operation == "delete" and not target:
        console.print("❌ [red]Delete nécessite --target[/red]")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("✏️ Édition intelligente...", total=None)
        
        # Édition
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            edit_file_with_ai(
                file_path,
                operation,
                target=target,
                new_content=new_content,
                line_number=line_number
            )
        )
        loop.close()
        
        progress.update(task, completed=100)
    
    if result["success"]:
        console.print("✅ [green]Édition réussie![/green]")
        
        # Changements
        changes = result["changes"]
        changes_table = Table(title="📊 Modifications")
        changes_table.add_column("Métrique", style="blue")
        changes_table.add_column("Valeur", style="green")
        
        changes_table.add_row("📄 Lignes ajoutées", str(changes['lines_added']))
        changes_table.add_row("✏️ Caractères ajoutés", str(changes['chars_added']))
        
        if result.get("backup_path") and backup:
            changes_table.add_row("💾 Backup", Path(result["backup_path"]).name)
        
        console.print(changes_table)
        
        # Diff preview
        if changes.get("diff_preview"):
            console.print("\n🔍 [bold]Aperçu des changements:[/bold]")
            diff_syntax = Syntax(changes["diff_preview"], "diff", theme="monokai")
            console.print(diff_syntax)
    
    else:
        console.print(f"❌ [red]Erreur édition: {result.get('error', 'Erreur inconnue')}[/red]")
        
        if result.get("backup_restored"):
            console.print("🔄 [yellow]Fichier restauré depuis backup[/yellow]")
        
        raise typer.Exit(1)

@app.command()
def create(
    file_path: str = typer.Argument(..., help="Chemin du fichier à créer"),
    template_type: str = typer.Argument(..., help="Type: class, function, api, component, test"),
    name: str = typer.Option(..., "--name", "-n", help="Nom de l'élément"),
    description: str = typer.Option("", "--desc", "-d", help="Description"),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Aperçu avant création")
):
    """
    🏗️ Créer un fichier avec template IA ultra-avancé
    """
    print_header()
    
    valid_templates = ["class", "function", "api", "component", "test"]
    if template_type not in valid_templates:
        console.print(f"❌ [red]Template invalide. Utilisez: {', '.join(valid_templates)}[/red]")
        raise typer.Exit(1)
    
    # Contexte template
    context = {
        "name": name,
        "description": description or f"{template_type.title()} généré par ESERISIA AI"
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("🏗️ Génération template ultra-avancé...", total=None)
        
        # Création
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            create_file_with_template(file_path, template_type, context)
        )
        loop.close()
        
        progress.update(task, completed=100)
    
    if result["success"]:
        console.print("✅ [green]Fichier créé avec succès![/green]")
        
        # Informations fichier
        info_table = Table(title="📄 Fichier Créé")
        info_table.add_column("Propriété", style="blue")
        info_table.add_column("Valeur", style="green")
        
        info_table.add_row("📁 Fichier", result["file_path"])
        info_table.add_row("🏗️ Template", result["template_type"])
        info_table.add_row("🔤 Langage", result["language"].title())
        
        # Stats
        analysis = result["analysis"]
        info_table.add_row("📄 Lignes", str(analysis.lines))
        info_table.add_row("🔧 Complexité", analysis.complexity)
        
        console.print(info_table)
        
        # Aperçu
        if preview:
            console.print(f"\n👀 [bold]Aperçu ({Path(file_path).name}):[/bold]")
            
            content_preview = result["content_preview"]
            syntax = Syntax(content_preview, result["language"], theme="monokai", line_numbers=True)
            console.print(syntax)
    
    else:
        console.print("❌ [red]Erreur création fichier[/red]")
        raise typer.Exit(1)

@app.command()
def status():
    """
    📊 Afficher le status complet de l'IDE ESERISIA AI
    """
    print_header()
    
    status_data = get_ide_capabilities()
    
    # Informations principales
    main_table = Table(title="🚀 Status ESERISIA IDE")
    main_table.add_column("Propriété", style="blue")
    main_table.add_column("Valeur", style="green")
    
    main_table.add_row("🧠 Système", status_data["system"])
    main_table.add_row("🔢 Version", status_data["version"])
    main_table.add_row("🎯 Précision", status_data["precision"])
    main_table.add_row("📁 Projet", status_data["project_path"])
    main_table.add_row("🎯 Mission", status_data["mission"])
    main_table.add_row("✅ Status", status_data["status"])
    
    console.print(main_table)
    
    # Capacités
    console.print("\n⚡ [bold blue]Capacités Ultra-Avancées:[/bold blue]")
    for capability in status_data["capabilities"]:
        console.print(f"  ✅ {capability}")
    
    # Performance
    perf_table = Table(title="🚀 Performance")
    perf_table.add_column("Métrique", style="cyan")
    perf_table.add_column("Valeur", style="green")
    
    for key, value in status_data["performance"].items():
        perf_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(perf_table)
    
    # Langages supportés
    console.print(f"\n🔤 [bold]Langages Supportés ({len(set(status_data['supported_languages']))}):[/bold]")
    
    languages = sorted(set(status_data["supported_languages"]))
    for i, lang in enumerate(languages):
        if i % 5 == 0:
            console.print()
        console.print(f"  • {lang.title():<12}", end="")
    
    console.print("\n")

@app.command()
def web(
    port: int = typer.Option(8506, "--port", "-p", help="Port pour interface web")
):
    """
    🌐 Lancer l'interface web IDE ultra-avancée
    """
    print_header()
    
    console.print(f"🚀 [green]Lancement interface web sur port {port}...[/green]")
    
    # Lancement Streamlit
    import subprocess
    
    web_app_path = Path(__file__).parent.parent / "web" / "app_ide.py"
    
    try:
        cmd = [
            "streamlit", "run", str(web_app_path),
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        console.print(f"📍 Interface disponible sur: http://localhost:{port}")
        console.print("🔥 [bold]ESERISIA IDE Ultra-Avancé est maintenant opérationnel![/bold]")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        console.print("\n👋 [yellow]Interface fermée[/yellow]")
    except Exception as e:
        console.print(f"❌ [red]Erreur lancement: {e}[/red]")

@app.command()
def evolve():
    """
    🧠 Déclencher évolution intelligence ESERISIA AI
    """
    print_header()
    
    console.print("🧠 [bold blue]Évolution Intelligence ESERISIA en cours...[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("🧠 Auto-évolution IA...", total=None)
        
        # Déclencher évolution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(eserisia_ide.evolve_now())
        loop.close()
        
        progress.update(task, completed=100)
    
    if success:
        console.print("✅ [green]Évolution intelligence réussie![/green]")
        
        # Afficher nouveau status
        console.print("\n📊 [bold]Nouveau Status Intelligence:[/bold]")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        evolution_status = loop.run_until_complete(eserisia_ide.get_evolution_intelligence_status())
        loop.close()
        
        if "precision" in evolution_status:
            console.print(f"🎯 Précision: {evolution_status['precision']}")
            console.print(f"🧠 Niveau: {evolution_status.get('intelligence_level', 'N/A')}")
            
            if "database_size" in evolution_status:
                db_stats = evolution_status["database_size"]
                console.print(f"📊 Événements: {db_stats.get('events', 0)}")
                console.print(f"🧩 Patterns: {db_stats.get('patterns', 0)}")
                console.print(f"⚡ Optimisations: {db_stats.get('optimizations', 0)}")
        
        console.print("\n💡 [italic]L'IA continue d'évoluer automatiquement à chaque utilisation[/italic]")
        
    else:
        console.print("❌ [red]Erreur lors de l'évolution[/red]")
        console.print("💡 [yellow]Vérifiez que la base de données est disponible[/yellow]")

@app.command()
def intelligence():
    """
    📊 Afficher status détaillé de l'intelligence évolutive
    """
    print_header()
    
    console.print("📊 [bold blue]Status Intelligence Évolutive ESERISIA[/bold blue]\n")
    
    # Récupérer status intelligence
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    evolution_status = loop.run_until_complete(eserisia_ide.get_evolution_intelligence_status())
    loop.close()
    
    if "evolution_enabled" in evolution_status and not evolution_status["evolution_enabled"]:
        console.print("⚠️ [yellow]Évolution non disponible[/yellow]")
        console.print(f"Raison: {evolution_status.get('reason', 'N/A')}")
        console.print("\n💡 Installez les dépendances base de données:")
        console.print("   pip install aiosqlite aioredis")
        return
    
    # Informations principales
    info_table = Table(title="🧠 Intelligence ESERISIA")
    info_table.add_column("Métrique", style="blue")
    info_table.add_column("Valeur", style="green")
    
    if "precision" in evolution_status:
        info_table.add_row("🎯 Précision", evolution_status["precision"])
    
    if "intelligence_level" in evolution_status:
        level = evolution_status["intelligence_level"]
        description = evolution_status.get("level_description", "")
        info_table.add_row("🧠 Niveau Intelligence", f"{level:.2f} - {description}")
    
    if "learning_enabled" in evolution_status:
        info_table.add_row("📚 Apprentissage", "✅ Actif" if evolution_status["learning_enabled"] else "❌ Inactif")
    
    if "database_path" in evolution_status:
        info_table.add_row("💾 Base de Données", evolution_status["database_path"])
    
    if "redis_enabled" in evolution_status:
        info_table.add_row("⚡ Cache Redis", "✅ Actif" if evolution_status["redis_enabled"] else "❌ Non disponible")
    
    console.print(info_table)
    
    # Statistiques base de données
    if "database_size" in evolution_status:
        db_stats = evolution_status["database_size"]
        
        stats_table = Table(title="📊 Statistiques Apprentissage")
        stats_table.add_column("Type", style="cyan")
        stats_table.add_column("Nombre", style="green")
        
        stats_table.add_row("📝 Événements d'analyse", str(db_stats.get("events", 0)))
        stats_table.add_row("🧩 Patterns appris", str(db_stats.get("patterns", 0)))
        stats_table.add_row("⚡ Optimisations réussies", str(db_stats.get("optimizations", 0)))
        
        console.print(stats_table)
    
    # Métriques évolution
    if "evolution_metrics" in evolution_status:
        metrics = evolution_status["evolution_metrics"]
        
        console.print(f"\n📈 [bold]Évolution sur 30 jours:[/bold]")
        console.print(f"  📝 Événements: {metrics.get('total_learning_events', 0)}")
        console.print(f"  ⚡ Optimisations: {metrics.get('successful_optimizations', 0)}")
        console.print(f"  🧩 Nouveaux patterns: {metrics.get('patterns_discovered', 0)}")
        console.print(f"  📊 Amélioration moyenne: {metrics.get('average_improvement', 0):.2f}%")
        
        if metrics.get("learning_rate"):
            console.print(f"  🧠 Taux apprentissage: {metrics['learning_rate']:.2f}%")
    
    # Prochaine évolution
    if "next_evolution" in evolution_status:
        console.print(f"\n🔮 [bold]Prochaine évolution:[/bold] {evolution_status['next_evolution']}")
    
    console.print("\n💡 [italic]Utilisez 'eserisia_ide evolve' pour déclencher une évolution manuelle[/italic]")

@app.command("generate")
def generate_project(
    project_type: str = typer.Argument(..., help="Type de projet: web, api, ml, mobile, blockchain, game, desktop, devops"),
    name: str = typer.Argument(..., help="Nom du projet"),
    path: str = typer.Option(".", help="Chemin où créer le projet"),
    template: Optional[str] = typer.Option(None, help="Template spécifique à utiliser"),
    config_file: Optional[str] = typer.Option(None, help="Fichier de configuration JSON")
):
    """🚀 Génère un projet complet ultra-avancé"""
    console.print(Panel.fit(
        f"[bold blue]🤖 ESERISIA AI - GÉNÉRATION DE PROJET ULTRA-AVANCÉE[/bold blue]\n"
        f"[green]Type:[/green] {project_type}\n"
        f"[green]Nom:[/green] {name}\n"
        f"[green]Chemin:[/green] {path}",
        border_style="blue"
    ))
    
    # Configuration du projet
    config = {}
    if config_file and Path(config_file).exists():
        with open(config_file) as f:
            config = json.load(f)
    
    # Configuration interactive si nécessaire
    if not config:
        config = _interactive_project_config(project_type, name)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"Génération du projet {name}...", total=None)
        
        try:
            # Génération du projet
            result = asyncio.run(project_generator.generate_project(
                project_type=project_type,
                project_name=name,
                destination_path=path,
                template_name=template,
                config=config
            ))
            
            progress.update(task, description="✅ Projet généré avec succès!")
            
            # Affichage du résultat
            _display_project_generation_result(result)
            
        except Exception as e:
            progress.update(task, description=f"❌ Erreur: {str(e)}")
            console.print(f"[red]❌ Erreur lors de la génération: {e}[/red]")
            raise typer.Exit(code=1)

@app.command("templates")
def list_templates(
    project_type: Optional[str] = typer.Option(None, help="Filtrer par type de projet")
):
    """📋 Liste tous les templates disponibles"""
    templates = project_generator.list_available_templates()
    
    if project_type:
        templates = {k: v for k, v in templates.items() if k == project_type}
    
    console.print(Panel.fit(
        "[bold blue]🎨 TEMPLATES DISPONIBLES - ESERISIA AI[/bold blue]",
        border_style="blue"
    ))
    
    for ptype, template_list in templates.items():
        console.print(f"\n[bold green]📁 {ptype.upper()}[/bold green]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Nom", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Technologies", style="yellow")
        
        for template in template_list:
            tech_str = ", ".join(template.get('technologies', []))
            table.add_row(
                template['name'],
                template['description'],
                tech_str
            )
        
        console.print(table)

def _interactive_project_config(project_type: str, project_name: str) -> Dict[str, Any]:
    """Configuration interactive du projet"""
    config = {
        "project_name": project_name,
        "project_type": project_type
    }
    
    console.print(f"\n[bold yellow]⚙️  Configuration du projet {project_type}[/bold yellow]")
    
    # Configuration spécifique par type
    if project_type == "web":
        config["frontend_framework"] = typer.prompt("Framework frontend", default="next")
        config["use_typescript"] = typer.confirm("Utiliser TypeScript ?", default=True)
        config["use_tailwind"] = typer.confirm("Utiliser Tailwind CSS ?", default=True)
        config["database"] = typer.prompt("Base de données", default="postgresql")
        
    elif project_type == "api":
        config["framework"] = typer.prompt("Framework API", default="fastapi")
        config["database"] = typer.prompt("Base de données", default="postgresql")
        config["use_redis"] = typer.confirm("Utiliser Redis ?", default=True)
        config["auth_type"] = typer.prompt("Type d'authentification", default="jwt")
        
    elif project_type == "ml":
        config["ml_framework"] = typer.prompt("Framework ML", default="pytorch")
        config["use_gpu"] = typer.confirm("Support GPU ?", default=True)
        config["include_mlflow"] = typer.confirm("Inclure MLflow ?", default=True)
        config["data_format"] = typer.prompt("Format de données", default="csv")
        
    elif project_type == "mobile":
        config["platform"] = typer.prompt("Plateforme", default="react-native")
        config["use_expo"] = typer.confirm("Utiliser Expo ?", default=True)
        config["navigation"] = typer.prompt("Navigation", default="react-navigation")
        
    elif project_type == "blockchain":
        config["blockchain"] = typer.prompt("Blockchain", default="ethereum")
        config["framework"] = typer.prompt("Framework", default="hardhat")
        config["use_openzeppelin"] = typer.confirm("Utiliser OpenZeppelin ?", default=True)
        
    elif project_type == "game":
        config["engine"] = typer.prompt("Moteur de jeu", default="unity")
        config["platform"] = typer.prompt("Plateforme cible", default="pc")
        config["graphics"] = typer.prompt("Type de graphismes", default="3d")
        
    elif project_type == "desktop":
        config["framework"] = typer.prompt("Framework desktop", default="electron")
        config["ui_library"] = typer.prompt("Bibliothèque UI", default="react")
        
    elif project_type == "devops":
        config["cloud_provider"] = typer.prompt("Fournisseur cloud", default="aws")
        config["container_orchestration"] = typer.prompt("Orchestration", default="kubernetes")
        config["ci_cd"] = typer.prompt("CI/CD", default="github-actions")
    
    return config

def _display_project_generation_result(result: Dict[str, Any]):
    """Affiche le résultat de la génération de projet"""
    console.print(f"\n[bold green]✅ PROJET GÉNÉRÉ AVEC SUCCÈS![/bold green]")
    
    # Informations du projet
    info_table = Table(show_header=False)
    info_table.add_column("Propriété", style="cyan")
    info_table.add_column("Valeur", style="white")
    
    info_table.add_row("📁 Nom", result.get('project_name', 'N/A'))
    info_table.add_row("📂 Type", result.get('project_type', 'N/A'))
    info_table.add_row("📍 Chemin", str(result.get('project_path', 'N/A')))
    info_table.add_row("🎨 Template", result.get('template_used', 'N/A'))
    info_table.add_row("📄 Fichiers créés", str(result.get('files_created', 0)))
    
    console.print(Panel(info_table, title="📊 Informations du projet", border_style="green"))
    
    # Commandes de démarrage
    if 'setup_commands' in result:
        console.print(f"\n[bold yellow]🚀 COMMANDES DE DÉMARRAGE[/bold yellow]")
        for i, cmd in enumerate(result['setup_commands'], 1):
            console.print(f"[cyan]{i}.[/cyan] {cmd}")
    
    # Structure du projet
    if 'project_structure' in result:
        console.print(f"\n[bold blue]📁 STRUCTURE DU PROJET[/bold blue]")
        tree = Tree(result['project_name'])
        _build_project_tree(tree, result['project_structure'])
        console.print(tree)
    
    console.print(f"\n[green]✨ Projet prêt à utiliser dans: {result.get('project_path')}[/green]")

def _build_project_tree(tree, structure, path=""):
    """Construit l'arbre de la structure du projet"""
    if isinstance(structure, dict):
        for name, content in structure.items():
            if isinstance(content, dict):
                branch = tree.add(f"📁 {name}")
                _build_project_tree(branch, content, f"{path}/{name}")
            else:
                tree.add(f"📄 {name}")
    elif isinstance(structure, list):
        for item in structure:
            tree.add(f"📄 {item}")

if __name__ == "__main__":
    app()
