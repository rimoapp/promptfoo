import { Command } from 'commander';
import * as fs from 'fs';
import * as path from 'path';
import { getDirectory } from '../esm';
import logger from '../logger';

export function setupVersionCommand(program: Command): void {
  program.option('--version', 'Print version', () => {
    const packageJson = JSON.parse(
      fs.readFileSync(path.join(getDirectory(), '../package.json'), 'utf8'),
    );
    logger.info(packageJson.version);
    process.exit(0);
  });
}
