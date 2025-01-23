from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Start command handler
async def start(update: Update, context):
    """Handles the /start command by sending a welcome message."""
    await update.message.reply_text("Hello! Send me a message, and I'll respond.")

# Message handler (responding based on user input)
async def handle_message(update: Update, context):
    """Processes user messages and provides appropriate responses."""
    user_message = update.message.text.strip()  # Remove leading/trailing whitespace

    # Example questions and responses
    # if "hi" in user_message:
    #     response = "Hi! Welcome back!"
    # elif "how are you" in user_message:
    #     response = "I'm doing great, thanks! How about you?"
    # elif "what is your name" in user_message:
    #     response = "I'm a Telegram bot. What's your name?"
    # elif "what can you do" in user_message:
    #     response = "I can respond to your messages! Try asking me something simple. :)"
    # elif "that's too bad" in user_message:
    #     response = "Yes, I know. I'm sorry to hear that..."
    # elif "see you soon" in user_message:
    #     response = "Yes, see you soon!"
    # else:
    #     # If the bot doesn't understand, repeat the original message
    #     response = f"Sorry, I don't understand that question. You said: '{update.message.text}'"


    messages = [
        {"role": "system", "content": "You are a friendly chatbot who is an expert on animals."},
        {"role": "user", "content": user_message},
    ]
    
    # Apply chat template to format the input for the model
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Generate response
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=200,  # Limit response length
        num_return_sequences=1,  # Generate one response
        temperature=0.7,  # Control randomness
        top_k=50,  # Use top-k sampling
        top_p=0.9,  # Use nucleus sampling
        do_sample=True  # Enable sampling for randomness
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the bot's response
    bot_response = generated_text.split(user_message, 1)[-1].strip()  # Extract text after user's message
    bot_response = bot_response.replace("<|assistant|>", "").strip()  # Remove any leftover tokens

    # Log the input and response for debugging
    print(f"Message received: {user_message}")  # Log the user's message
    print(f"Bot response: {bot_response}")  # Log the bot's response

    # Send the cleaned response to the user
    await update.message.reply_text(bot_response)

# Main function to run the bot
def main():
    """Initializes and starts the Telegram bot."""
    # BotFather token
    app = ApplicationBuilder().token("8087807001:AAHVhksm1ysTu4DY2V-WLXtogwEHO9qxvPA").build()

    # Register command and message handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
