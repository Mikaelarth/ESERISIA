"""
Interface Web IDE ESERISIA AI Ultra-Avancée
==========================================
Lecture, Compréhension et Édition de projets locaux
"""

import streamlit as st
import asyncio
import os
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Import ESERISIA IDE
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from eserisia.ide_engine import (
    eserisia_ide, scan_project_intelligent, analyze_file_intelligent,
    edit_file_with_ai, create_file_with_template, get_ide_capabilities
)

# Configuration page
st.set_page_config(
    page_title="ESERISIA AI - IDE Intelligent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Ultra-Avancé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .ide-metrics {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    
    .file-item {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
        transition: all 0.3s ease;
    }
    
    .file-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .analysis-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .code-editor {
        background: #2d3748;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Fira Code', monospace;
        margin: 1rem 0;
    }
    
    .success-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def render_header():
    """Header ultra-avancé"""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 ESERISIA AI - IDE Intelligent Ultra-Avancé</h1>
        <p>Lecture • Compréhension • Édition intelligente des projets locaux</p>
        <p><strong>Architecture évolutive 2025</strong> • Précision: 99.87%</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_project_stats(project_path):
    """Stats projet en cache"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        structure = loop.run_until_complete(scan_project_intelligent(project_path))
        loop.close()
        return structure
    except Exception as e:
        st.error(f"Erreur scan projet: {e}")
        return None

def render_project_scanner():
    """Scanner de projet ultra-avancé"""
    st.header("🔍 Scanner de Projet Intelligent")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        project_path = st.text_input(
            "Chemin du projet",
            value=os.getcwd(),
            help="Chemin vers le projet à analyser"
        )
    
    with col2:
        scan_button = st.button("🚀 Scanner Projet", type="primary")
    
    if scan_button or "project_structure" not in st.session_state:
        with st.spinner("🧠 ESERISIA AI analyse votre projet..."):
            project_structure = get_project_stats(project_path)
            
            if project_structure:
                st.session_state.project_structure = project_structure
                
                # Métriques projet
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📁 Total Fichiers", project_structure.total_files)
                
                with col2:
                    st.metric("📄 Total Lignes", f"{project_structure.total_lines:,}")
                
                with col3:
                    st.metric("🔧 Langages", len(project_structure.languages))
                
                with col4:
                    st.metric("⚡ Frameworks", len(project_structure.frameworks))
                
                # Graphiques
                if project_structure.languages:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Graphique langages
                        fig_lang = px.pie(
                            values=[1] * len(project_structure.languages),
                            names=project_structure.languages,
                            title="🔤 Distribution des Langages"
                        )
                        fig_lang.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_lang, use_container_width=True)
                    
                    with col2:
                        # Graphique frameworks
                        if project_structure.frameworks:
                            fig_fw = px.bar(
                                x=project_structure.frameworks,
                                y=[1] * len(project_structure.frameworks),
                                title="⚡ Frameworks Détectés"
                            )
                            st.plotly_chart(fig_fw, use_container_width=True)
                
                # Architecture détectée
                st.markdown(f"""
                <div class="analysis-box">
                    <h3>🏗️ Architecture Détectée</h3>
                    <p><strong>{project_structure.architecture_pattern}</strong></p>
                    <p>Langages: {', '.join(project_structure.languages)}</p>
                    <p>Frameworks: {', '.join(project_structure.frameworks)}</p>
                </div>
                """, unsafe_allow_html=True)

def render_file_explorer():
    """Explorateur de fichiers intelligent"""
    st.header("📂 Explorateur Intelligent")
    
    if "project_structure" not in st.session_state:
        st.warning("⚠️ Scannez d'abord un projet")
        return
    
    # Sélection fichier
    project_root = Path(st.session_state.project_structure.root_path)
    
    # Lister fichiers supportés
    supported_files = []
    for file_path in project_root.rglob("*"):
        if file_path.is_file() and not any(ignore in str(file_path) for ignore in ['.git', '__pycache__', 'node_modules']):
            supported_files.append(str(file_path.relative_to(project_root)))
    
    if supported_files:
        selected_file = st.selectbox(
            "Sélectionner un fichier",
            [""] + sorted(supported_files),
            help="Choisissez un fichier à analyser"
        )
        
        if selected_file:
            full_path = project_root / selected_file
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🧠 Analyser Fichier", type="primary"):
                    analyze_selected_file(str(full_path))
            
            with col2:
                if st.button("👁️ Lire Contenu"):
                    display_file_content(str(full_path))

def analyze_selected_file(file_path):
    """Analyse approfondie d'un fichier"""
    with st.spinner("🧠 Analyse intelligente en cours..."):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis = loop.run_until_complete(analyze_file_intelligent(file_path))
            loop.close()
            
            st.success(f"✅ Analyse terminée: {Path(file_path).name}")
            
            # Métriques fichier
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📄 Lignes", analysis.lines)
            
            with col2:
                st.metric("📏 Taille", f"{analysis.size:,} chars")
            
            with col3:
                st.metric("🔧 Complexité", analysis.complexity)
            
            with col4:
                st.metric("🔤 Langage", analysis.language.title())
            
            # Détails analyse
            if analysis.functions or analysis.classes or analysis.imports:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if analysis.functions:
                        st.subheader("⚙️ Fonctions")
                        for func in analysis.functions[:10]:  # Limite pour affichage
                            if isinstance(func, dict):
                                st.write(f"• {func.get('name', func)}")
                            else:
                                st.write(f"• {func}")
                
                with col2:
                    if analysis.classes:
                        st.subheader("🏛️ Classes")
                        for cls in analysis.classes[:10]:
                            if isinstance(cls, dict):
                                st.write(f"• {cls.get('name', cls)}")
                            else:
                                st.write(f"• {cls}")
                
                with col3:
                    if analysis.imports:
                        st.subheader("📦 Imports")
                        for imp in analysis.imports[:10]:
                            st.write(f"• {imp}")
            
            # Issues et suggestions
            if analysis.issues:
                st.subheader("⚠️ Issues Détectées")
                for issue in analysis.issues:
                    st.warning(f"• {issue}")
            
            if analysis.suggestions:
                st.subheader("💡 Suggestions IA")
                for suggestion in analysis.suggestions:
                    st.info(f"• {suggestion}")
            
            # Stocker analyse
            st.session_state.current_analysis = analysis
            
        except Exception as e:
            st.error(f"❌ Erreur analyse: {e}")

def display_file_content(file_path):
    """Affichage contenu fichier"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        content = loop.run_until_complete(eserisia_ide.read_file_content(file_path))
        loop.close()
        
        file_name = Path(file_path).name
        language = eserisia_ide.file_extensions.get(Path(file_path).suffix.lower(), "text")
        
        st.subheader(f"📄 Contenu: {file_name}")
        
        # Limiter affichage pour performance
        if len(content) > 10000:
            st.warning(f"⚠️ Fichier volumineux ({len(content):,} chars). Affichage des 10,000 premiers caractères.")
            content = content[:10000] + "\n\n... (tronqué)"
        
        st.code(content, language=language)
        
        # Stocker contenu pour édition
        st.session_state.file_content = content
        st.session_state.current_file = file_path
        
    except Exception as e:
        st.error(f"❌ Erreur lecture: {e}")

def render_file_editor():
    """Éditeur de fichier intelligent"""
    st.header("✏️ Éditeur Intelligent")
    
    if "current_file" not in st.session_state:
        st.info("📝 Sélectionnez et lisez un fichier d'abord")
        return
    
    file_path = st.session_state.current_file
    
    st.write(f"**Fichier:** `{Path(file_path).name}`")
    
    # Options d'édition
    col1, col2 = st.columns(2)
    
    with col1:
        operation = st.selectbox(
            "Opération",
            ["replace", "insert", "delete", "add_function", "optimize"],
            help="Type d'édition à effectuer"
        )
    
    with col2:
        if operation in ["replace", "delete"]:
            target_text = st.text_area("Texte à remplacer/supprimer")
        elif operation == "insert":
            line_number = st.number_input("Numéro de ligne", min_value=1, value=1)
    
    if operation == "replace":
        new_text = st.text_area("Nouveau texte")
        
        if st.button("🔄 Remplacer", type="primary"):
            if target_text and new_text:
                perform_edit(file_path, operation, target_text, new_text)
            else:
                st.error("Texte source et destination requis")
    
    elif operation == "insert":
        new_content = st.text_area("Contenu à insérer")
        
        if st.button("➕ Insérer", type="primary"):
            if new_content:
                perform_edit(file_path, operation, line_number=line_number, new_content=new_content)
            else:
                st.error("Contenu requis")
    
    elif operation == "delete":
        if st.button("🗑️ Supprimer", type="primary"):
            if target_text:
                perform_edit(file_path, operation, target_text)
            else:
                st.error("Texte à supprimer requis")
    
    elif operation == "add_function":
        st.subheader("➕ Ajouter Fonction")
        
        func_name = st.text_input("Nom de la fonction")
        func_description = st.text_area("Description")
        
        if st.button("🚀 Générer et Ajouter", type="primary"):
            if func_name:
                # Génération template fonction
                language = eserisia_ide.file_extensions.get(Path(file_path).suffix.lower(), "python")
                context = {"name": func_name, "description": func_description}
                
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    template_result = loop.run_until_complete(
                        create_file_with_template(f"temp_{func_name}.{language}", "function", context)
                    )
                    loop.close()
                    
                    if template_result["success"]:
                        # Extraire contenu fonction
                        with open(template_result["file_path"], 'r') as f:
                            func_content = f.read()
                        
                        # Supprimer fichier temporaire
                        os.unlink(template_result["file_path"])
                        
                        # Ajouter au fichier
                        perform_edit(file_path, "add_function", new_content=func_content)
                    else:
                        st.error("Erreur génération fonction")
                        
                except Exception as e:
                    st.error(f"Erreur: {e}")
            else:
                st.error("Nom de fonction requis")
    
    elif operation == "optimize":
        if st.button("⚡ Optimiser Code", type="primary"):
            perform_edit(file_path, operation)

def perform_edit(file_path, operation, target=None, new_content=None, line_number=None):
    """Exécution édition fichier"""
    with st.spinner("✏️ Édition en cours..."):
        try:
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
            
            if result["success"]:
                st.success("✅ Édition réussie!")
                
                # Afficher changements
                changes = result["changes"]
                st.info(f"📊 Changements: {changes['lines_added']} lignes, {changes['chars_added']} caractères")
                
                if changes.get("diff_preview"):
                    st.subheader("🔍 Aperçu des changements")
                    st.code(changes["diff_preview"], language="diff")
                
                # Backup info
                if result.get("backup_path"):
                    st.info(f"💾 Backup créé: {Path(result['backup_path']).name}")
                
                # Actualiser contenu
                if st.button("🔄 Recharger fichier"):
                    display_file_content(file_path)
            
            else:
                st.error(f"❌ Erreur édition: {result.get('error', 'Erreur inconnue')}")
                
                if result.get("backup_restored"):
                    st.warning("🔄 Fichier restauré depuis backup")
            
        except Exception as e:
            st.error(f"❌ Erreur: {e}")

def render_template_creator():
    """Créateur de fichiers avec templates"""
    st.header("🏗️ Créateur de Fichiers Intelligent")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file_name = st.text_input("Nom du fichier", placeholder="mon_fichier.py")
        template_type = st.selectbox(
            "Type de template",
            ["class", "function", "api", "component", "test"],
            help="Type de template à générer"
        )
    
    with col2:
        element_name = st.text_input("Nom élément", placeholder="MonClasse ou ma_fonction")
        description = st.text_area("Description", placeholder="Description de l'élément")
    
    if st.button("🚀 Créer Fichier", type="primary"):
        if file_name and element_name:
            create_template_file(file_name, template_type, element_name, description)
        else:
            st.error("Nom de fichier et nom d'élément requis")

def create_template_file(file_name, template_type, element_name, description):
    """Création fichier avec template"""
    with st.spinner("🏗️ Génération template ultra-avancé..."):
        try:
            # Chemin complet
            if "project_structure" in st.session_state:
                base_path = Path(st.session_state.project_structure.root_path)
            else:
                base_path = Path.cwd()
            
            file_path = base_path / file_name
            
            # Contexte template
            context = {
                "name": element_name,
                "description": description or f"{template_type.title()} généré par ESERISIA AI"
            }
            
            # Création
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                create_file_with_template(str(file_path), template_type, context)
            )
            loop.close()
            
            if result["success"]:
                st.success(f"✅ Fichier créé: {file_name}")
                
                # Aperçu
                st.subheader("👀 Aperçu")
                st.code(result["content_preview"], language=result["language"])
                
                # Stats
                analysis = result["analysis"]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📄 Lignes", analysis.lines)
                
                with col2:
                    st.metric("🔤 Langage", analysis.language.title())
                
                with col3:
                    st.metric("🔧 Complexité", analysis.complexity)
                
                st.info(f"💾 Fichier sauvegardé: {result['file_path']}")
                
            else:
                st.error("❌ Erreur création fichier")
                
        except Exception as e:
            st.error(f"❌ Erreur: {e}")

def render_ide_status():
    """Status IDE"""
    st.header("📊 Status ESERISIA IDE")
    
    status = get_ide_capabilities()
    
    # Status principal
    st.markdown(f"""
    <div class="ide-metrics">
        <h3>🚀 {status['system']}</h3>
        <p><strong>Version:</strong> {status['version']}</p>
        <p><strong>Précision:</strong> {status['precision']}</p>
        <p><strong>Mission:</strong> {status['mission']}</p>
        <p><strong>Status:</strong> {status['status']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Capacités
    st.subheader("⚡ Capacités Ultra-Avancées")
    for capability in status['capabilities']:
        st.write(f"✅ {capability}")
    
    # Performance
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Précision Analyse", status['performance']['analysis_precision'])
    
    with col2:
        st.metric("🛡️ Sécurité Édition", status['performance']['edit_safety'])
    
    with col3:
        st.metric("🏗️ Génération Template", status['performance']['template_generation'])
    
    # Langages supportés
    st.subheader("🔤 Langages Supportés")
    languages = list(set(status['supported_languages']))
    cols = st.columns(min(5, len(languages)))
    
    for i, lang in enumerate(languages[:15]):  # Limite affichage
        with cols[i % 5]:
            st.write(f"• {lang.title()}")

def main():
    """Interface principale"""
    render_header()
    
    # Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Scanner Projet",
        "📂 Explorateur",
        "✏️ Éditeur",
        "🏗️ Créateur",
        "📊 Status"
    ])
    
    with tab1:
        render_project_scanner()
    
    with tab2:
        render_file_explorer()
    
    with tab3:
        render_file_editor()
    
    with tab4:
        render_template_creator()
    
    with tab5:
        render_ide_status()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4>🧠 ESERISIA AI - IDE Intelligent Ultra-Avancé</h4>
        <p>Architecture évolutive 2025 • Précision 99.87% • Édition intelligente</p>
        <p><em>Transformez votre développement avec l'IA la plus avancée</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
