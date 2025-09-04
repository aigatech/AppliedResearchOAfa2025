import React, { useState } from 'react';
import './index.css';

function App() {
  const [pantry, setPantry] = useState('');
  const [diet, setDiet] = useState('none');
  const [time, setTime] = useState(30);
  const [recipe, setRecipe] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const generateRecipe = async (e) => {
    e.preventDefault();
    
    if (!pantry.trim()) {
      setError('Please enter some ingredients in your pantry');
      return;
    }

    setLoading(true);
    setError('');
    setRecipe(null);

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pantry: pantry.trim(),
          diet: diet,
          time: parseInt(time)
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate recipe');
      }

      setRecipe(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatIngredients = (ingredients) => {
    if (Array.isArray(ingredients)) {
      return ingredients;
    }
    if (typeof ingredients === 'string') {
      return ingredients.split(',').map(ing => ing.trim()).filter(ing => ing);
    }
    return [];
  };

  const formatInstructions = (instructions) => {
    let cleanInstructions = [];
    
    if (Array.isArray(instructions)) {
      for (let instruction of instructions) {
        if (typeof instruction === 'string') {
          // Skip empty strings, single words, or malformed entries
          const cleaned = instruction.trim();
          if (cleaned && 
              cleaned.length > 10 && 
              !cleaned.startsWith('"RecipeInstructions"') &&
              !cleaned.includes('RecipeIngredientParts') &&
              cleaned !== 'Madison' &&
              !cleaned.match(/^[A-Z][a-z]+$/) && // Skip single proper nouns
              cleaned.includes(' ')) { // Must contain spaces (actual sentences)
            cleanInstructions.push(cleaned);
          }
        }
      }
    } else if (typeof instructions === 'string') {
      // Try to split by numbered steps or sentences
      const steps = instructions.split(/\d+\.|\./).map(step => step.trim()).filter(step => step);
      cleanInstructions = steps.filter(step => 
        step.length > 10 && 
        step.includes(' ') &&
        !step.startsWith('"RecipeInstructions"')
      );
    }
    
    return cleanInstructions;
  };

  return (
    <div className="container">
      <div className="header">
        <h1>🍳 Recipe Generator</h1>
        <p>Enter your pantry ingredients and get a personalized recipe!</p>
      </div>

      <form onSubmit={generateRecipe} className="form-section">
        <div className="form-group">
          <label htmlFor="pantry">
            Pantry Ingredients (comma-separated)
          </label>
          <textarea
            id="pantry"
            value={pantry}
            onChange={(e) => setPantry(e.target.value)}
            placeholder="e.g., pasta, tomato, garlic, onion, basil, olive oil, parmesan"
            rows="3"
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="diet">Dietary Restrictions</label>
            <select
              id="diet"
              value={diet}
              onChange={(e) => setDiet(e.target.value)}
            >
              <option value="none">None</option>
              <option value="vegetarian">Vegetarian</option>
              <option value="vegan">Vegan</option>
              <option value="gluten-free">Gluten-Free</option>
              <option value="keto">Keto</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="time">Cooking Time (minutes)</label>
            <input
              type="number"
              id="time"
              value={time}
              onChange={(e) => setTime(e.target.value)}
              min="5"
              max="180"
            />
          </div>
        </div>

        <button
          type="submit"
          className="generate-btn"
          disabled={loading || !pantry.trim()}
        >
          {loading ? 'Generating Recipe...' : 'Generate Recipe'}
        </button>
      </form>

      {loading && (
        <div className="loading">
          <div>Generating your recipe...</div>
        </div>
      )}

      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {recipe && (
        <div className="recipe-result">
          <h2>{recipe.Name || 'Generated Recipe'}</h2>
          
          {recipe.RecipeIngredientParts && (
            <div>
              <h3>Ingredients</h3>
              <ul className="ingredients-list">
                {formatIngredients(recipe.RecipeIngredientParts).map((ingredient, index) => (
                  <li key={index}>{ingredient}</li>
                ))}
              </ul>
            </div>
          )}

          {recipe.RecipeInstructions && (
            <div>
              <h3>Instructions</h3>
              {formatInstructions(recipe.RecipeInstructions).length > 0 ? (
                <ol className="instructions-list">
                  {formatInstructions(recipe.RecipeInstructions).map((instruction, index) => (
                    <li key={index}>{instruction}</li>
                  ))}
                </ol>
              ) : (
                <div style={{
                  padding: '15px',
                  backgroundColor: '#fff3cd',
                  border: '1px solid #ffeaa7',
                  borderRadius: '8px',
                  color: '#856404'
                }}>
                  <p><strong>Note:</strong> The recipe instructions could not be properly formatted. Here's the raw data:</p>
                  <p style={{ fontFamily: 'monospace', fontSize: '14px' }}>
                    {JSON.stringify(recipe.RecipeInstructions, null, 2)}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
