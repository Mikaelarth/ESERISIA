export default function Home() {
  return (
    <main className="min-h-screen p-8">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-4xl font-bold mb-4 text-blue-600">
          🚀 mon-projet-web
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          Propulsé par ESERISIA AI - L'IDE le plus puissant au monde
        </p>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg">
          <h2 className="text-2xl font-semibold mb-4">✨ Fonctionnalités</h2>
          <ul className="space-y-2">
            <li>⚡ Next.js 14 avec App Router</li>
            <li>🎨 Tailwind CSS intégré</li>
            <li>📱 Responsive design</li>
            <li>🔒 TypeScript pour la sécurité</li>
          </ul>
        </div>
      </div>
    </main>
  )
}