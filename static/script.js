document.addEventListener('DOMContentLoaded', () => {
  const queryForm = document.getElementById('queryForm');
  const queryInput = document.getElementById('queryInput');
  const topKInput = document.getElementById('topKInput');
  const resultsContainer = document.getElementById('resultsContainer');
  const loadingElement = document.getElementById('loading');
  const contextsList = document.getElementById('contextsList');
  const answerContainer = document.getElementById('answerContainer');
  
  queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const query = queryInput.value.trim();
    const topK = parseInt(topKInput.value) || 3;
    
    if (!query) {
      alert('Please enter a question');
      return;
    }
    
    // Show loading spinner
    loadingElement.style.display = 'block';
    resultsContainer.classList.add('hidden');
    
    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          top_k: topK
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Clear previous results
      contextsList.innerHTML = '';
      
      // Display contexts
      data.contexts.forEach(context => {
        const contextItem = document.createElement('div');
        contextItem.className = 'context-item';
        
        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'context-score';
        scoreSpan.textContent = `${context.score}% match`;
        
        const textDiv = document.createElement('div');
        textDiv.className = 'context-text';
        textDiv.textContent = context.text;
        
        contextItem.appendChild(scoreSpan);
        contextItem.appendChild(textDiv);
        contextsList.appendChild(contextItem);
      });
      
      // Display answer
      document.getElementById('answerText').textContent = data.answer;
      
      // Show results
      resultsContainer.classList.remove('hidden');
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred while processing your request. Please try again.');
    } finally {
      // Hide loading spinner
      loadingElement.style.display = 'none';
    }
  });
  
  // Add ripple effect to buttons
  const buttons = document.querySelectorAll('.btn');
  buttons.forEach(button => {
    button.addEventListener('click', function(e) {
      const x = e.clientX - e.target.getBoundingClientRect().left;
      const y = e.clientY - e.target.getBoundingClientRect().top;
      
      const ripple = document.createElement('span');
      ripple.style.left = `${x}px`;
      ripple.style.top = `${y}px`;
      ripple.className = 'ripple';
      
      this.appendChild(ripple);
      
      setTimeout(() => {
        ripple.remove();
      }, 600);
    });
  });
}); 