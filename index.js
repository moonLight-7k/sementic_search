const fs = require('fs');
const cheerio = require('cheerio');
import("@xenova/transformers").then((transformers) => {
  ({ pipeline, env } = transformers);
  env.useBrowserCache = false;
  env.allowLocalModels = true;
});


// Dynamically import node-fetch
import("node-fetch").then((fetchModule) => {
  const fetch = fetchModule.default;

  // Initialize the embedding pipeline
  let embeddingPipeline = null;
  async function initializeEmbeddingPipeline() {
    if (!embeddingPipeline) {
      // Using MiniLM model which is smaller and faster than BERT but still effective
      embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
    return embeddingPipeline;
  }

  // Helper functions
  async function fetchWebpageContent(url) {
    try {
      const response = await fetch(url);
      const html = await response.text();
      const $ = cheerio.load(html);

      // Get title and description
      const title = $("title").text().trim() || $("h1").first().text().trim();
      const description = $('meta[name="description"]').attr("content") || "";
      const content = $("main, article, .content, #content")
        .text()
        .replace(/\s+/g, " ")
        .trim();

      return { title, content, description };
    } catch (error) {
      console.error(`Error fetching content for ${url}:`, error);
      return { title: '', content: '', description: '' };
    }
  }

  // Function to generate embedding for a given text
  async function generateEmbedding(text) {
    try {
      const pipeline = await initializeEmbeddingPipeline();

      // Truncate text if it's too long (model has max token limit)
      const truncatedText = text.slice(0, 512);

      // Generate embedding
      const result = await pipeline(truncatedText, {
        pooling: 'mean',
        normalize: true
      });

      // Convert to Array for easier handling
      return Array.from(result.data);
    } catch (error) {
      console.error("Error generating embedding:", error);
      throw error;
    }
  }

  // Cosine Similarity to compare embeddings
  function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) {
      throw new Error("Invalid vectors provided for comparison");
    }
    if (vecA.length !== vecB.length) {
      throw new Error("Vectors must be of the same length");
    }

    const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
  }

  // Enhance bookmark with content and embedding
  async function enhanceBookmark(bookmark) {
    try {

      const enhancedBookmark = { ...bookmark };

      // Fetch content if the site is not a local IP
      if (!bookmark.site.match(/^https?:\/\/(\d{1,3}\.){3}\d{1,3}/)) {
        const webContent = await fetchWebpageContent(bookmark.site);
        enhancedBookmark.title = webContent.title;
        enhancedBookmark.description = webContent.description;
        enhancedBookmark.content = webContent.content;
      }

      // Combine text for embedding (title, description, content, tags, categories)
      const textForEmbedding = [
        enhancedBookmark.title,
        enhancedBookmark.description,
        enhancedBookmark.content,
        ...enhancedBookmark.category,
        ...enhancedBookmark.tag
      ].filter(Boolean).join(" ");

      // Generate embedding if text is available
      if (textForEmbedding) {
        enhancedBookmark.embedding = await generateEmbedding(textForEmbedding);
      }

      return enhancedBookmark;
    } catch (error) {
      console.error(`Error enhancing bookmark ${bookmark.site}:`, error);
    }

  }

  // Perform semantic search
  async function semanticSearch(query, bookmarks, limit = 5) {
    try {
      const queryEmbedding = await generateEmbedding(query);

      const results = bookmarks
        .filter(bookmark => bookmark.embedding)
        .map(bookmark => ({
          ...bookmark,
          similarity: cosineSimilarity(queryEmbedding, bookmark.embedding)
        }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, limit);

      return results;
    } catch (error) {
      console.error("Error during semantic search:", error);
      throw error;
    }
  }

  // Load bookmarks from file
  async function loadBookmarksFromFile(filePath) {
    if (fs.existsSync(filePath)) {
      const fileContent = fs.readFileSync(filePath, "utf-8");
      return JSON.parse(fileContent);
    } else {
      console.error(`${filePath} not found`);
      return [];
    }
  }

  // Save enhanced bookmarks to file
  function saveEnhancedBookmarks(bookmarks, outputFilePath) {
    fs.writeFileSync(outputFilePath, JSON.stringify(bookmarks, null, 2));
  }

  // Main function to enhance bookmarks and perform search
  async function main() {
    try {
      const embeddingFilePath = 'embedding.json'; // Output file for embeddings
      const inputFilePath = 'site.json'; // Path to your site.json file
      const outputFilePath = 'enhanced_bookmarks.json'; // Output file for enhanced bookmarks

      // Load bookmarks
      const bookmarks = await loadBookmarksFromFile(inputFilePath);

      // Create an array to hold embeddings
      const embeddingArray = [];

      // Enhance bookmarks with title, content, and embedding
      const enhancedBookmarks = [];
      for (let bookmark of bookmarks) {
        const enhancedBookmark = await enhanceBookmark(bookmark);
        enhancedBookmarks.push(enhancedBookmark);

        // Add the embedding to the embeddings array
        if (enhancedBookmark.embedding) {
          embeddingArray.push(enhancedBookmark.embedding);
        }

        console.log(`Enhanced bookmark: ${bookmark.site}`);
      }

      // Save enhanced bookmarks to a new file
      saveEnhancedBookmarks(enhancedBookmarks, outputFilePath);

      // Save embeddings to a separate file
      saveEmbeddings(embeddingArray, embeddingFilePath);  // Now using embeddingArray

      // Perform semantic search with a sample query
      const query = "icon";
      const searchResults = await semanticSearch(query, enhancedBookmarks);

      // Display search results
      console.log("\nSearch Results:");
      searchResults.forEach((result, index) => {
        console.log(`${index + 1}. ${result.title || result.site}`);
        console.log(`   URL: ${result.site}`);
        console.log(`   Similarity: ${(result.similarity * 100).toFixed(2)}%`);
        if (result.description) {
          console.log(`   Description: ${result.description}`);
        }
      });
    } catch (error) {
      console.error("Error in main function:", error);
    }
  }


  // Run the main function
  main();

  // Save embeddings to a separate file
  function saveEmbeddings(embeddings, outputFilePath) {
    fs.writeFileSync(outputFilePath, JSON.stringify(embeddings, null, 2));
  }
});



