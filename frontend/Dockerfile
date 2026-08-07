# Build context for this Dockerfile is frontend/ itself.
#   docker build -f frontend/Dockerfile -t vessel-frontend ./frontend
FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
