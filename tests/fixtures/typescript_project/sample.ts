// tests/fixtures/typescript_project/sample.ts
import { readFileSync } from "fs";
import path from "path";

export function greet(name: string): string {
    return `Hello, ${name}!`;
}

export class Formatter {
    private prefix: string;

    constructor(prefix: string) {
        this.prefix = prefix;
    }

    format(value: string): string {
        return `${this.prefix}: ${value}`;
    }
}
